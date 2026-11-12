"""Post-hoc feasibility comparisons on the reproduced 20-world synthetic study.

Run after the existing confirmation runner:
    python tools/evaluate_confirmation_projection.py

No neural fits, hyperparameter choices, or original artifacts are modified.
All additional intervals are descriptive whole-world bootstrap intervals.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from linn.evaluation.metrics import regression_metrics, workstation_violations
from linn.evaluation.projection import project_durations, clip_durations
from tools.evaluate_projection_baselines import feature_matrix, ridge_predictions

TARGETS = ['elapsed_hours', 'booked_hours']
METHODS = ['ridge', 'ridge_projection', 'ridge_clipping',
           'data_driven', 'data_driven_projection', 'data_driven_clipping',
           'domain_informed', 'domain_informed_projection', 'domain_informed_clipping']
COMPARISONS = (
    [('domain_informed', baseline) for baseline in
     ['data_driven', 'data_driven_projection', 'ridge', 'ridge_projection',
      'data_driven_clipping', 'ridge_clipping']]
    + [('domain_informed_projection', baseline) for baseline in
       ['data_driven_projection', 'ridge_projection', 'data_driven_clipping', 'ridge_clipping']]
)


def summarize_paired(metrics, repetitions=20000, seed=61031):
    """Average paired workstation effects within worlds before resampling.

    Averages of relative effects are intentionally distinct from ratios of
    average errors. A world, rather than a workstation or job, is one draw.
    """
    keys = ['world', 'machine', 'completeness', 'target']
    if metrics.duplicated(keys + ['method']).any():
        raise ValueError('Duplicate paired metric rows')
    paired = metrics.pivot(index=keys, columns='method', values='MAE')
    rows = []
    for candidate, baseline in COMPARISONS:
        if not {candidate, baseline}.issubset(paired.columns):
            raise ValueError('Missing comparator method')
        values = paired[[candidate, baseline]]
        if not np.isfinite(values.to_numpy()).all() or (values[baseline] <= 0).any():
            raise ValueError('Each paired baseline must have a finite positive MAE')
        effect = values.reset_index()[keys].copy()
        effect['candidate'] = candidate
        effect['baseline'] = baseline
        effect['absolute_improvement_hours'] = (values[baseline] - values[candidate]).to_numpy()
        effect['relative_improvement'] = (1 - values[candidate] / values[baseline]).to_numpy()
        rows.append(effect)
    machine_effects = pd.concat(rows, ignore_index=True)
    group_keys = ['world', 'completeness', 'target', 'candidate', 'baseline']
    world_effects = machine_effects.groupby(group_keys, sort=True)[
        ['absolute_improvement_hours', 'relative_improvement']].mean().reset_index()
    intervals = []
    for group, values in world_effects.groupby(group_keys[1:], sort=True):
        values = values.sort_values('world')
        if len(values) < 2:
            raise ValueError('At least two independent worlds are needed')
        draws = np.random.default_rng(seed).integers(0, len(values), (repetitions, len(values)))
        for metric in ['absolute_improvement_hours', 'relative_improvement']:
            effects = values[metric].to_numpy()
            low, high = np.quantile(effects[draws].mean(axis=1), [.025, .975])
            intervals.append({**dict(zip(group_keys[1:], group)), 'metric': metric,
                              'mean': float(effects.mean()), 'ci_low': float(low),
                              'ci_high': float(high), 'world_sd': float(effects.std(ddof=1)),
                              'n_worlds': len(effects), 'positive_worlds': int((effects > 0).sum()),
                              'bootstrap_repetitions': repetitions,
                              'interpretation': 'Descriptive post-hoc comparison; no multiplicity correction'})
    return machine_effects, world_effects, pd.DataFrame(intervals)


def run(root=ROOT):
    root = Path(root)
    source = root / 'results/synthetic_confirmation'
    output = root / 'results/confirmation_projection'
    settings_path = root / 'configs/synthetic_confirmation.json'
    settings = json.loads(settings_path.read_text())
    hashes = {}

    def record(path):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    record(settings_path)
    for path in [source / 'protocol.json', source / 'environment.json', source / 'validation.json',
                 root / 'docs/synthetic_confirmation_protocol.md',
                 root / 'tools/evaluate_projection_baselines.py',
                 root / 'src/linn/evaluation/projection.py', Path(__file__).resolve()]:
        record(path)
    validation = json.loads((source / 'validation.json').read_text())
    if validation.get('fits') != 240 or not validation.get('protected_unchanged'):
        raise ValueError('The complete, protected 240-fit reproduction must finish first')

    def read(path, **kwargs):
        return pd.read_csv(record(path), **kwargs)

    rows = []
    for world in settings['simulation_seeds']:
        frame = read(source / 'references' / f'world_{world}.csv',
                     parse_dates=['start'], dtype={'operation': str})
        for machine, group in frame.groupby('machine', sort=True):
            train = group[group.split.eq('train')]
            test = group[group.split.eq('test')]
            for level in settings['label_completeness']:
                key = f'{world}_{machine}_{int(level*100):03d}'
                audit = read(source / 'conditions' / f'{key}_mask.csv').set_index('record_id')
                mask = audit.loc[train.record_id, ['elapsed_observed', 'booked_observed']].to_numpy(dtype=bool)
                n_validation = audit.loc[audit.split.eq('validation'),
                                         ['elapsed_observed', 'booked_observed']].sum().to_numpy(dtype=int)
                expected_train = int(360*level)
                expected_validation = int(120*level)
                np.testing.assert_array_equal(mask.sum(axis=0), [expected_train]*2)
                np.testing.assert_array_equal(n_validation, [expected_validation]*2)
                metadata = json.loads(record(source / 'conditions' / f'{key}_preprocessing.json').read_text())
                encoder = metadata['encoder']
                if encoder['fitted_record_ids'] != train.record_id.tolist():
                    raise ValueError('Archived feature fitting rows differ from training rows')
                x = feature_matrix(train, train, encoder)
                test_x = feature_matrix(test, train, encoder)
                y = train[TARGETS].to_numpy(dtype=np.float32)
                y[~mask] = np.nan
                predictions = {'ridge': ridge_predictions(x, y, mask, test_x, alpha=1.)}
                observed = None
                for method in ['data_driven', 'domain_informed']:
                    saved = read(source / 'runs' / f'{key}_{method}_predictions.csv')
                    if saved.record_id.tolist() != test.record_id.tolist():
                        raise ValueError('Saved predictions and test records are misaligned')
                    current = saved[['elapsed_observed', 'booked_observed']].to_numpy(dtype=float)
                    np.testing.assert_allclose(current, test[TARGETS].to_numpy(dtype=np.float32),
                                               rtol=2e-7, atol=1e-7)
                    if observed is not None:
                        np.testing.assert_array_equal(current, observed)
                    observed = current
                    predictions[method] = saved[['elapsed_predicted', 'booked_predicted']].to_numpy(dtype=float)
                for method in ['ridge', 'data_driven', 'domain_informed']:
                    predictions[method + '_projection'] = project_durations(predictions[method], 0.)
                    predictions[method + '_clipping'] = clip_durations(predictions[method], 0.)
                for method in METHODS:
                    p = predictions[method]
                    violations = workstation_violations(p, 0.)
                    if method.endswith(('_projection', '_clipping')):
                        if violations['any_violation_fraction'] != 0.:
                            raise AssertionError('Output correction left a constraint violation')
                    for j, target in enumerate(TARGETS):
                        rows.append({'world': world, 'machine': machine, 'completeness': level,
                                     'method': method, 'target': target, 'n_test': len(test),
                                     'n_train_observed': int(mask[:, j].sum()),
                                     'n_validation_observed': int(n_validation[j]),
                                     'constraint_violation': violations['any_violation_fraction'],
                                     **regression_metrics(observed[:, j], p[:, j])})

    metrics = pd.DataFrame(rows)
    expected_rows = len(settings['simulation_seeds']) * 3 * len(settings['label_completeness']) * len(METHODS) * 2
    if len(metrics) != expected_rows:
        raise ValueError('Incomplete comparator metrics')
    world_metrics = metrics.groupby(['world', 'completeness', 'target', 'method'], sort=True)[
        ['MAE', 'RMSE', 'R2', 'constraint_violation']].mean().reset_index()
    aggregate = world_metrics.groupby(['completeness', 'target', 'method'], sort=True).agg(
        MAE=('MAE', 'mean'), MAE_world_sd=('MAE', 'std'), RMSE=('RMSE', 'mean'),
        R2=('R2', 'mean'), constraint_violation=('constraint_violation', 'mean')).reset_index()
    machine_effects, world_effects, intervals = summarize_paired(
        metrics, repetitions=settings['bootstrap_replicates'], seed=settings['bootstrap_seed'])
    for path, expected in hashes.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise AssertionError(f'Input changed during analysis: {path}')
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [('metrics', metrics), ('world_metrics', world_metrics),
                        ('aggregate_metrics', aggregate), ('paired_machine_effects', machine_effects),
                        ('paired_world_effects', world_effects), ('paired_intervals_descriptive', intervals)]:
        table.to_csv(output / f'{name}.csv', index=False)
    paper_methods = {'ridge': 'Ridge', 'ridge_projection': 'Ridge + Projection',
                     'data_driven': 'Data-driven NN', 'data_driven_projection': 'Data-driven NN + Projection',
                     'domain_informed': 'Domain-informed NN',
                     'domain_informed_projection': 'Domain-informed NN + Projection'}
    tex = [r'\begin{table}[t]', r'\centering\small',
           r'\caption{Post-hoc comparisons on the reproduced 20-world synthetic design. '
           r'Training and validation labels are both reduced. Values average three '
           r'workstations equally within each world and then average 20 worlds. '
           r'Duration errors are in hours. Additional comparator analyses are descriptive.}',
           r'\label{tab:confirmation_projection}', r'\begin{tabular}{llrrr}', r'\toprule',
           r'Labels & Method & Elapsed MAE & Booked MAE & Violations (\%) \\', r'\midrule']
    for level in settings['label_completeness']:
        for method, name in paper_methods.items():
            g = aggregate[(aggregate.completeness == level) & (aggregate.method == method)].set_index('target')
            tex.append(f'{100*level:.0f}\\% & {name} & {g.loc["elapsed_hours", "MAE"]:.4f} & '
                       f'{g.loc["booked_hours", "MAE"]:.4f} & '
                       f'{100*g.loc["elapsed_hours", "constraint_violation"]:.2f} ' + r'\\')
        if level != settings['label_completeness'][-1]:
            tex.append(r'\midrule')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (output / 'paper_table.tex').write_text('\n'.join(tex), encoding='utf-8')
    manifest = {
        'analysis': 'Post-hoc output correction and fixed-alpha Ridge on reproduced confirmation worlds',
        'interpretation': 'Exploratory descriptive comparisons, not preregistered primary tests',
        'reproduction_provenance': 'Original protocol unavailable; execution document reconstructed from existing code/configuration on 2026-11-10',
        'source': 'results/synthetic_confirmation', 'independent_worlds': len(settings['simulation_seeds']),
        'ridge_alpha': 1., 'ridge_intercept': 'Unpenalized', 'tolerance_hours': 0.,
        'features': 'Same training-fitted encoders and archived per-target masks as neural study',
        'checkpoint_policy': 'Existing neural predictions reused without checkpoint reselection',
        'projection': 'Equal-weight Euclidean distance in original hours onto d>=0,b>=0,b<=d',
        'clipping': 'd=max(d,0); b=min(max(b,0),d)',
        'bootstrap_unit': 'Complete world after averaging paired workstation effects',
        'bootstrap_repetitions': settings['bootstrap_replicates'], 'bootstrap_seed': settings['bootstrap_seed'],
        'source_artifacts_unchanged': True, 'input_sha256': hashes,
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas']}}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    summary = ['# Post-hoc feasibility comparisons on reproduced confirmation worlds', '',
               'These are added descriptive analyses of the existing 20-world design. The original '
               'protocol document was missing; its execution specification was reconstructed before '
               'the present rerun. The previously reported primary outcome was already known. '
               'This is a reproduction and post-hoc extension, not new prospective confirmation.', '',
               'Both training and validation labels are reduced: at 5%, each target has 18 training '
               'and six validation labels per workstation. Ridge uses exactly the archived training '
               'masks and training-fitted feature representation, fixed alpha 1 and an unpenalized '
               'intercept. Neural predictions and checkpoints are unchanged. Both output corrections '
               'yield zero violations; projection may alter elapsed as well as booked time.', '',
               '## Macro metrics', '',
               '| Labels | Method | Elapsed MAE (h) | Booked MAE (h) | Joint violations (%) |',
               '|---:|---|---:|---:|---:|']
    for level in settings['label_completeness']:
        for method in METHODS:
            g = aggregate[(aggregate.completeness == level) & (aggregate.method == method)].set_index('target')
            summary.append(f'| {level:.0%} | {method} | {g.loc["elapsed_hours", "MAE"]:.6f} | '
                           f'{g.loc["booked_hours", "MAE"]:.6f} | '
                           f'{100*g.loc["elapsed_hours", "constraint_violation"]:.3f} |')
    summary += ['', '## Descriptive paired elapsed-MAE reductions at 5% labels', '',
                'Effects first average paired relative workstation reductions within a world, then '
                'average worlds. They need not equal ratios of macro MAEs. Intervals resample '
                '20 complete worlds 20,000 times; no multiplicity correction is applied.', '',
                '| Candidate | Baseline | Mean reduction (%) | Approximate 95% interval (%) | Positive worlds |',
                '|---|---|---:|---|---:|']
    selected = intervals[(intervals.completeness == .05) & (intervals.target == 'elapsed_hours') &
                         intervals.metric.eq('relative_improvement')]
    for row in selected.itertuples(index=False):
        summary.append(f'| {row.candidate} | {row.baseline} | {100*row.mean:.2f} | '
                       f'[{100*row.ci_low:.2f}, {100*row.ci_high:.2f}] | {row.positive_worlds}/20 |')
    summary += ['', 'Ridge regularization was fixed rather than optimized. The generator family and '
                'workstation parameters are unchanged; these results do not establish industrial '
                'generalization. Comparing projected predictors on both sides is necessary when '
                'deployment requires guaranteed feasibility. Projection guarantees feasibility, '
                'not an improvement in every target-specific MAE.', '',
                'Reproduce with `python tools/evaluate_confirmation_projection.py` after the '
                'confirmation runner finishes. All input SHA-256 hashes and additional intervals '
                'are retained in this directory. No original source artifacts are modified.', '']
    (output / 'summary.md').write_text('\n'.join(summary), encoding='utf-8')
    print('\n'.join(summary))


if __name__ == '__main__':
    run()
