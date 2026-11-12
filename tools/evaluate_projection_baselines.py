"""Post-hoc projection comparison on saved synthetic degradation predictions.

Run from the repository root: python tools/evaluate_projection_baselines.py
Only NumPy and pandas are needed. No neural fits or original outputs are changed.
"""
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from linn.evaluation.metrics import regression_metrics, workstation_violations
from linn.evaluation.projection import project_durations, clip_durations

TARGETS = ['elapsed_hours', 'booked_hours']
METHODS = ['ridge', 'ridge_projection', 'data_driven', 'data_driven_projection',
           'domain_informed', 'ridge_clipping', 'data_driven_clipping']


def feature_matrix(frame, train, saved_encoder):
    """Recreate the frozen train-only encoder without sklearn dependencies.

    Calendar and quantity scaler statistics are recalculated from training rows
    and checked against the archived encoder; no target values enter encoding.
    The cast to float32 matches WorkstationDataset feature construction.
    """
    values = {}
    for col in ['product', 'operation']:
        for category in sorted(train[col].fillna('__missing__').astype(str).unique()):
            values[f'categories__{col}_{category}'] = (
                frame[col].fillna('__missing__').astype(str).eq(category).to_numpy(dtype=float))
    def calendar(part):
        hour = part.start.dt.hour + part.start.dt.minute / 60
        day = part.start.dt.dayofweek
        return np.column_stack([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                                np.sin(2*np.pi*day/7), np.cos(2*np.pi*day/7)])
    raw = calendar(train)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale == 0] = 1.
    np.testing.assert_allclose(mean, saved_encoder['calendar_mean'], atol=1e-12)
    np.testing.assert_allclose(scale, saved_encoder['calendar_scale'], atol=1e-12)
    normalized = (calendar(frame) - mean) / scale
    for i, name in enumerate(['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']):
        values[f'calendar__{name}'] = normalized[:, i]
    raw = np.column_stack([train.planned_quantity.to_numpy(dtype=float), np.zeros(len(train))])
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale == 0] = 1.
    np.testing.assert_allclose(mean, saved_encoder['numeric_mean'], atol=1e-12)
    np.testing.assert_allclose(scale, saved_encoder['numeric_scale'], atol=1e-12)
    normalized = (np.column_stack([frame.planned_quantity.to_numpy(dtype=float),
                                   np.zeros(len(frame))]) - mean) / scale
    values['numeric__planned_quantity'] = normalized[:, 0]
    values['numeric__quantity_missing'] = normalized[:, 1]
    names = saved_encoder['feature_names']
    if set(names) != set(values):
        raise ValueError('Archived feature names do not match reconstructed features')
    return np.column_stack([values[name] for name in names]).astype(np.float32)


def ridge_predictions(x, y, mask, test_x, alpha=1.):
    """Independent masked ridge targets; unpenalized intercept; fixed alpha.

    Minimize sum((y - X beta - intercept)**2) + alpha * sum(beta**2).
    Hidden y entries are excluded before any calculation. Float64 SVD is used
    to solve this same standard ridge objective without installing sklearn.
    """
    if alpha <= 0:
        raise ValueError('Ridge alpha must be positive')
    columns = []
    for j in range(2):
        keep = np.asarray(mask[:, j], dtype=bool)
        if not keep.any():
            raise ValueError('Each target requires observed training labels')
        observed_x = np.asarray(x[keep], dtype=float)
        observed_y = np.asarray(y[keep, j], dtype=float)
        if not np.isfinite(observed_y).all():
            raise ValueError('Observed labels must be finite')
        x_mean, y_mean = observed_x.mean(axis=0), observed_y.mean()
        u, singular, vt = np.linalg.svd(observed_x - x_mean, full_matrices=False)
        coef = vt.T @ ((singular / (singular**2 + alpha)) * (u.T @ (observed_y - y_mean)))
        columns.append((test_x - x_mean) @ coef + y_mean)
    return np.column_stack(columns)


def run(root=ROOT):
    source = root / 'results/synthetic_degradation'
    out = root / 'results/projection_baselines'
    out.mkdir(parents=True, exist_ok=True)
    config_path = root / 'configs/synthetic_degradation.json'
    settings = json.loads(config_path.read_text())
    provenance = [config_path, root / 'tools/evaluate_projection_baselines.py',
                  root / 'src/linn/evaluation/projection.py', root / 'src/linn/evaluation/metrics.py']
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in provenance}
    rows = []
    def read(path, **kwargs):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path, **kwargs)
    for world in settings['simulation_seeds']:
        frame = read(source / 'references' / f'world_{world}.csv',
                     parse_dates=['start'], dtype={'operation': str})
        for machine, group in frame.groupby('machine', sort=True):
            train = group[group.split.eq('train')]
            test = group[group.split.eq('test')]
            for seed in settings['paired_seeds']:
                for level in settings['label_completeness']:
                    key = f'{world}_{machine}_{seed}_labels_{int(level*100):03d}'
                    audit = read(source / 'conditions' / f'{key}_mask.csv')
                    audit = audit.set_index('record_id').loc[train.record_id]
                    mask = audit[['elapsed_observed', 'booked_observed']].to_numpy(dtype=bool)
                    meta_path = source / 'conditions' / f'{key}_preprocessing.json'
                    hashes[str(meta_path.relative_to(root))] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
                    encoder = json.loads(meta_path.read_text())['encoder']
                    if encoder['fitted_record_ids'] != train.record_id.tolist():
                        raise ValueError('Archived encoder training rows differ')
                    x = feature_matrix(train, train, encoder)
                    test_x = feature_matrix(test, train, encoder)
                    y = train[TARGETS].to_numpy(dtype=np.float32)
                    y[~mask] = np.nan
                    ridge = ridge_predictions(x, y, mask, test_x)
                    predictions = {'ridge': ridge, 'ridge_projection': project_durations(ridge),
                                   'ridge_clipping': clip_durations(ridge)}
                    observed = None
                    for method in ['data_driven', 'domain_informed']:
                        saved = read(source / 'runs' / f'{key}_{method}_predictions.csv')
                        if saved.record_id.tolist() != test.record_id.tolist():
                            raise ValueError('Saved test predictions are not aligned')
                        current = saved[['elapsed_observed', 'booked_observed']].to_numpy(dtype=float)
                        np.testing.assert_allclose(current, test[TARGETS].to_numpy(dtype=np.float32),
                                                   rtol=2e-7, atol=1e-7)
                        if observed is not None:
                            np.testing.assert_array_equal(current, observed)
                        observed = current
                        p = saved[['elapsed_predicted', 'booked_predicted']].to_numpy(dtype=float)
                        predictions[method] = p
                        if method == 'data_driven':
                            predictions[method + '_projection'] = project_durations(p)
                            predictions[method + '_clipping'] = clip_durations(p)
                    for method in METHODS:
                        p = predictions[method]
                        violations = workstation_violations(p, tolerance=0.)
                        if method.endswith(('_projection', '_clipping')):
                            assert violations['any_violation_fraction'] == 0.
                        for j, target in enumerate(TARGETS):
                            rows.append({'world': world, 'machine': machine, 'seed': seed,
                                         'completeness': level, 'method': method, 'target': target,
                                         'n_train_observed': int(mask[:, j].sum()), 'n_test': len(test),
                                         'constraint_violation': violations['any_violation_fraction'],
                                         **regression_metrics(observed[:, j], p[:, j])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'metrics.csv', index=False)
    world_metrics = metrics.groupby(['world', 'completeness', 'method', 'target'], sort=True)[
        ['MAE', 'RMSE', 'R2', 'constraint_violation']].mean().reset_index()
    world_metrics.to_csv(out / 'world_metrics.csv', index=False)
    aggregate = world_metrics.groupby(['completeness', 'method', 'target'], sort=True).agg(
        MAE=('MAE', 'mean'), MAE_world_sd=('MAE', 'std'), RMSE=('RMSE', 'mean'),
        R2=('R2', 'mean'), constraint_violation=('constraint_violation', 'mean')).reset_index()
    aggregate.to_csv(out / 'aggregate_metrics.csv', index=False)
    paired = metrics.pivot(index=['world', 'machine', 'seed', 'completeness', 'target'],
                           columns='method', values='MAE').reset_index()
    for comparator in ['data_driven_projection', 'ridge_projection']:
        paired[f'reduction_vs_{comparator}'] = 1 - paired.domain_informed / paired[comparator]
    paired.to_csv(out / 'paired_effects.csv', index=False)
    effect_summary = paired.groupby(['completeness', 'target'])[[
        'reduction_vs_data_driven_projection', 'reduction_vs_ridge_projection']].mean().reset_index()
    effect_summary.to_csv(out / 'paired_effect_summary.csv', index=False)
    for relative, expected in hashes.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    manifest = {'analysis': 'Exploratory post-hoc reanalysis; not a new confirmatory study',
                'source': 'results/synthetic_degradation', 'ridge_alpha': 1.,
                'projection': 'Equal-hour Euclidean projection onto d>=0,b>=0,b<=d (epsilon=0)',
                'clipping': 'd=max(d,0); b=min(max(b,0),d)',
                'train_labels': 'Identical archived per-target training observation masks',
                'validation_labels': '120 complete validation labels/target/machine for existing neural fits',
                'ridge_selection': 'No tuning; alpha=1 fixed before computing comparison',
                'independent_worlds': len(settings['simulation_seeds']),
                'pairs_per_label_level': len(settings['simulation_seeds'])*3*len(settings['paired_seeds']),
                'source_artifacts_unchanged': True,
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'pandas': pd.__version__},
                'input_and_source_sha256': hashes}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    selected = aggregate[aggregate.completeness.isin([1., .1, .05])]
    selected.to_csv(out / 'paper_metrics.csv', index=False)
    names = {'ridge': 'Ridge', 'ridge_projection': 'Ridge + Projection',
             'data_driven': 'Data-driven NN', 'data_driven_projection': 'Data-driven NN + Projection',
             'domain_informed': 'Domain-informed NN'}
    tex = [r'\begin{table}[t]', r'\centering',
           r'\caption{Post-hoc feasibility baselines in the exploratory synthetic degradation study. '
           r'Macro MAE in hours and joint violations over three worlds, three machines and two paired '
           r'seeds. Training labels are reduced; validation labels remain complete. Projection is '
           r'Euclidean in hours onto $d\geq0$, $b\geq0$, $b\leq d$.}',
           r'\label{tab:projection_baselines}', r'\begin{tabular}{llrrr}', r'\toprule',
           r'Labels & Method & MAE $d$ & MAE $b$ & Viol. (\%) \\', r'\midrule']
    for level in [1., .1, .05]:
        for method, label in names.items():
            g = selected[(selected.completeness == level) & (selected.method == method)].set_index('target')
            tex.append(f'{100*level:.0f}\\% & {label} & {g.loc["elapsed_hours", "MAE"]:.3f} & '
                       f'{g.loc["booked_hours", "MAE"]:.3f} & '
                       f'{100*g.loc["elapsed_hours", "constraint_violation"]:.1f} ' + r'\\')
        if level != .05:
            tex.append(r'\midrule')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (out / 'paper_table.tex').write_text('\n'.join(tex), encoding='utf-8')
    lines = ['# Post-hoc feasibility baselines', '',
             'This exploratory comparison reuses saved synthetic degradation predictions. '
             'It is separate from the 20-world confirmation; confirmation and industrial '
             'prediction artifacts are unavailable in this checkout. No original artifacts are changed.', '',
             'Three worlds, three machines and two paired mask/optimizer seeds give 18 paired '
             'runs at each training-label level. All neural fits used 120 complete validation '
             'labels per target and machine. The table averages runs equally; world SD first '
             'averages machines and seeds and is descriptive, not a confidence interval.', '',
             'Ridge uses the same train-only encoded features and exactly the archived '
             'target-channel masks, an unpenalized intercept and fixed alpha=1. It is fitted '
             'by float64 SVD; hidden labels are removed before fitting. No test labels enter '
             'features, fitting or parameter selection. Alpha=1 is a fixed comparator choice, '
             'not a validation-tuned optimum. Neural checkpoints remain those of the original study.', '',
             'Projection minimizes squared distance between duration pairs in original hours '
             'over d>=0, b>=0, b<=d (epsilon=0). It adjusts both channels when needed. '
             'The supplementary clipping sensitivity sets d=max(d,0) then '
             'b=min(max(b,0),d), preserving nonnegative elapsed predictions. '
             'All projected and clipped predictions have zero joint violations. '
             'Projection guarantees feasibility, not lower per-target MAE.', '',
             '| Labels | Method | Elapsed MAE (h) | Booked MAE (h) | Joint violation (%) |',
             '|---:|---|---:|---:|---:|']
    for level in [1., .1, .05]:
        for method in METHODS:
            g = selected[(selected.completeness == level) & (selected.method == method)].set_index('target')
            lines.append(f'| {level:.0%} | {method} | {g.loc["elapsed_hours", "MAE"]:.6f} | '
                         f'{g.loc["booked_hours", "MAE"]:.6f} | '
                         f'{100*g.loc["elapsed_hours", "constraint_violation"]:.3f} |')
    lines += ['', 'Paired relative reductions average per-run reductions, rather than ratios of aggregate MAEs.', '',
              '| Labels | Target | Domain-informed reduction vs NN + Projection | vs Ridge + Projection |',
              '|---:|---|---:|---:|']
    for row in effect_summary[effect_summary.completeness.isin([1., .1, .05])].itertuples(index=False):
        lines.append(f'| {row.completeness:.0%} | {row.target} | '
                     f'{row.reduction_vs_data_driven_projection:.2%} | '
                     f'{row.reduction_vs_ridge_projection:.2%} |')
    lines += ['', 'These descriptive comparisons use only three independently simulated worlds. '
              'They do not establish a general advantage over projection or industrial generalization. '
              'A soft penalty still leaves violations; applications requiring guaranteed feasibility '
              'must apply a correction or hard constraint regardless of predictive model.', '',
              'Reproduce with `python tools/evaluate_projection_baselines.py`. '
              'The input manifest records SHA-256 hashes and the script verifies inputs remain unchanged.', '']
    (out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    run()
