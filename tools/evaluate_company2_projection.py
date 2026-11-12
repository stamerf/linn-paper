"""Post-hoc corrections of the regenerated original Company 2 case predictions.

Run after the configured case experiment:
    .venv/Scripts/python tools/evaluate_company2_projection.py

Only aggregate outputs are written. Private record-level inputs remain in the
ignored original experiment directory; no neural or Ridge model is refitted.
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

LABELS = {'ridge': 'Ridge', 'ridge_projection': 'Ridge + Projection',
          'data_driven': 'Data-driven NN', 'data_driven_projection': 'Data-driven NN + Projection',
          'domain_informed': 'Domain-informed NN',
          'domain_informed_projection': 'Domain-informed NN + Projection',
          'ridge_clipping': 'Ridge + Clipping',
          'data_driven_clipping': 'Data-driven NN + Clipping'}


def analyze(root=ROOT):
    source = root / 'results/company2_case'
    out = root / 'results/company2_projection'
    settings = json.loads((root / 'configs/company2_case.json').read_text())
    tolerance = float(settings['rounding_tolerance_hours'])
    hashes, rows = {}, []
    def remember(path):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [root / 'configs/company2_case.json', root / 'tools/evaluate_company2_projection.py',
                 root / 'src/linn/evaluation/projection.py', root / 'src/linn/evaluation/metrics.py',
                 source / 'selected_configs.json', source / 'protocol.json', source / 'validation.json',
                 source / 'environment.json', source / 'raw_inventory.json', source / 'cohort_summary.json']:
        remember(path)
    validation = json.loads((source / 'validation.json').read_text())
    if not validation['protected_unchanged'] or not validation['primary_case_exploratory']:
        raise ValueError('Expected validated exploratory Company 2 case outputs')
    reference = None
    for level in settings['completeness']:
        for seed in settings['evaluation_seeds']:
            prefix = f'{int(100*level):03d}_{seed}_q1'
            observed = None
            predictions = {}
            for original, method in [('quantity_linear', 'ridge'), ('data_driven', 'data_driven'),
                                     ('domain_informed', 'domain_informed')]:
                path = source / 'runs' / f'{prefix}_{original}_predictions.csv'
                remember(path)
                frame = pd.read_csv(path)
                frame = frame[frame.split.eq('test')].copy()
                if frame.empty or frame.record_id.duplicated().any():
                    raise ValueError('Expected unique, nonempty held-out predictions')
                current = frame[['record_id', 'order_id', 'machine', 'elapsed_observed', 'booked_observed']]
                if observed is not None:
                    pd.testing.assert_frame_equal(current.reset_index(drop=True), observed.reset_index(drop=True))
                observed = current
                if reference is None:
                    reference = observed.copy()
                else:
                    pd.testing.assert_frame_equal(current.reset_index(drop=True), reference.reset_index(drop=True))
                p = frame[['elapsed_predicted', 'booked_predicted']].to_numpy(dtype=float)
                predictions[method] = p
                predictions[method + '_projection'] = project_durations(p, tolerance)
                if method != 'domain_informed':
                    predictions[method + '_clipping'] = clip_durations(p, tolerance)
            for method in LABELS:
                p = predictions[method]
                for machine in ['pooled'] + sorted(observed.machine.unique().tolist()):
                    keep = np.ones(len(observed), dtype=bool) if machine == 'pooled' else observed.machine.eq(machine)
                    subset = observed[keep]
                    v = workstation_violations(p[keep], tolerance)
                    if method.endswith(('_projection', '_clipping')):
                        assert v['any_violation_fraction'] == 0.
                    for j, (target, column) in enumerate([('elapsed_hours', 'elapsed_observed'),
                                                        ('booked_hours', 'booked_observed')]):
                        rows.append({'machine': machine, 'completeness': level, 'seed': seed,
                                     'method': method, 'target': target,
                                     'n_test': len(subset), 'n_orders': subset.order_id.nunique(),
                                     'constraint_violation': v['any_violation_fraction'],
                                     **regression_metrics(subset[column].to_numpy(), p[keep, j])})
    metrics = pd.DataFrame(rows)
    expected = len(settings['completeness'])*len(settings['evaluation_seeds'])*len(LABELS)*2
    assert len(metrics[metrics.machine.eq('pooled')]) == expected
    aggregate = metrics.groupby(['machine', 'completeness', 'method', 'target']).agg(
        MAE=('MAE', 'mean'), MAE_seed_sd=('MAE', 'std'), RMSE=('RMSE', 'mean'),
        R2=('R2', 'mean'), constraint_violation=('constraint_violation', 'mean'),
        n_test=('n_test', 'first'), n_seeds=('seed', 'nunique')).reset_index()
    paired = metrics[metrics.machine.eq('pooled')].pivot(
        index=['completeness', 'seed', 'target'], columns='method', values='MAE').reset_index()
    for comparator in ['data_driven', 'data_driven_projection', 'ridge_projection']:
        paired[f'reduction_vs_{comparator}'] = 1 - paired.domain_informed / paired[comparator]
        paired[f'projected_domain_reduction_vs_{comparator}'] = (
            1 - paired.domain_informed_projection / paired[comparator])
    effect_columns = [c for c in paired.columns
                      if c.startswith(('reduction_vs_', 'projected_domain_reduction_vs_'))]
    effects = paired.groupby(['completeness', 'target'])[effect_columns].agg(['mean', 'min', 'max'])
    effects.columns = ['_'.join(pair) for pair in effects.columns]
    effects = effects.reset_index()
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / 'metrics.csv', index=False)
    aggregate.to_csv(out / 'aggregate_metrics.csv', index=False)
    paired.to_csv(out / 'paired_effects.csv', index=False)
    effects.to_csv(out / 'paired_effect_summary.csv', index=False)
    observed_violation = workstation_violations(reference[['elapsed_observed', 'booked_observed']].to_numpy(), tolerance)
    for relative, digest in hashes.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest
    manifest = {'analysis': 'Post-hoc feasibility comparison on regenerated exploratory industrial case',
                'regeneration_provenance': 'Original case outputs were absent; existing configured '
                                          'experiment was re-executed on supplied raw files. '
                                          'This records current re-execution, not recovered historic freezing.',
                'source': 'results/company2_case', 'tolerance_hours': tolerance,
                'projection': 'Equal-hour Euclidean onto d>=0,b>=0,b<=d+epsilon',
                'clipping': 'd=max(d,0);b=min(max(b,0),d+epsilon)',
                'n_test_operations': len(reference), 'n_test_orders': int(reference.order_id.nunique()),
                'n_machines': int(reference.machine.nunique()),
                'observed_joint_violation_fraction': observed_violation['any_violation_fraction'],
                'source_artifacts_unchanged': True, 'seed_replication': 'Same test operations, repeated fitting/masks',
                'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
                'input_and_source_sha256': hashes}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    pooled = aggregate[aggregate.machine.eq('pooled')]
    lines = ['# Company 2: post-hoc feasibility baselines', '',
             'This comparison reads regenerated original Company 2 case predictions without refitting models. '
             f'It is exploratory and shares the original chronological test cohort. {len(settings["evaluation_seeds"])} seeds repeat fitting '
             'and target masks on these same operations, rather than providing independent industrial cohorts.', '',
             'The original case output directory was absent in this checkout. The existing configured '
             'case experiment, including complete-label configuration selection, was rerun on the supplied '
             'raw export before this comparison. New protocol and environment files document that '
             're-execution; they do not recover or establish an earlier freeze date.', '',
             f'The test cohort has {len(reference)} operation-machine aggregates and '
             f'{reference.order_id.nunique()} production orders across {reference.machine.nunique()} presses. '
             f'Observed test targets violate the joint accounting relationship on '
             f'{observed_violation["any_violation_fraction"]:.1%} of operations at the fixed '
             f'{tolerance:.3f} h tolerance. Corrections guarantee feasible predictions; they do not guarantee '
             'agreement with noisy or inconsistent observations.', '',
             'Ridge and the neural models use the original matched target masks, features and splits. '
             'Architectures and penalty weight were selected using complete development labels before '
             'reducing labels for the 25% and 10% fits. Sparse results retain this calibration advantage.', '',
             'Projection minimizes equal-weight squared adjustment in original hours over '
             'd>=0, b>=0, b<=d+epsilon and can adjust both outputs. The supplementary one-sided '
             'clipping caps booked time after clipping negative values; it preserves nonnegative '
             'elapsed predictions. Neither correction uses held-out observations.', '',
             '| Labels | Method | Elapsed MAE h | Booked MAE h | Elapsed R2 | Joint violation % |',
             '|---:|---|---:|---:|---:|---:|']
    tex = [r'\begin{table}[t]', r'\centering',
           r'\caption{Exploratory Company~2 post-hoc correction comparison. Means over '
           f'{len(settings["evaluation_seeds"])} fitting and mask seeds on the same '
           f'{len(reference)} held-out operations; MAEs in hours. The fixed accounting '
           r'tolerance is $\epsilon=0.02$\,h.}', r'\label{tab:industrial_projection}',
           r'\begin{tabular}{llrrrr}', r'\toprule',
           r'Labels & Method & MAE $d$ & MAE $b$ & $R_d^2$ & Viol. (\%) \\', r'\midrule']
    for level in sorted(settings['completeness'], reverse=True):
        for method, label in LABELS.items():
            g = pooled[pooled.completeness.eq(level) & pooled.method.eq(method)].set_index('target')
            d, b = g.loc['elapsed_hours'], g.loc['booked_hours']
            lines.append(f'| {level:.0%} | {label} | {d.MAE:.6f} | {b.MAE:.6f} | '
                         f'{d.R2:.6f} | {100*d.constraint_violation:.3f} |')
            if not method.endswith('_clipping'):
                tex.append(f'{100*level:.0f}\\% & {label} & {d.MAE:.3f} & {b.MAE:.3f} & '
                           f'{d.R2:.3f} & {100*d.constraint_violation:.1f} ' + r'\\')
        if level != min(settings['completeness']):
            tex.append(r'\midrule')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    lines += ['', 'Paired relative improvements average per-seed effects; positive means lower domain-informed MAE.', '',
              '| Labels | Target | vs NN | vs NN + Projection | vs Ridge + Projection |',
              '|---:|---|---:|---:|---:|']
    for _, row in effects.iterrows():
        lines.append(f'| {row.completeness:.0%} | {row.target} | '
                     f'{row.reduction_vs_data_driven_mean:+.2%} | '
                     f'{row.reduction_vs_data_driven_projection_mean:+.2%} | '
                     f'{row.reduction_vs_ridge_projection_mean:+.2%} |')
    lines += ['', 'Applying the same projection to the domain-informed model gives a comparison '
              'with feasible predictions on both sides:', '',
              '| Labels | Target | Projected domain reduction vs NN + Projection | vs Ridge + Projection |',
              '|---:|---|---:|---:|']
    for _, row in effects.iterrows():
        lines.append(f'| {row.completeness:.0%} | {row.target} | '
                     f'{row.projected_domain_reduction_vs_data_driven_projection_mean:+.2%} | '
                     f'{row.projected_domain_reduction_vs_ridge_projection_mean:+.2%} |')
    lines += ['', 'Seed standard deviations in `aggregate_metrics.csv` are descriptive. No new '
              'industrial significance or external generalization claim follows. Source predictions, '
              'selected settings and provenance files are hashed; no record identifiers are exported here.', '',
              'Reproduce after the original case run with '
              '`python tools/evaluate_company2_projection.py`.', '']
    (out / 'paper_table.tex').write_text('\n'.join(tex), encoding='utf-8')
    (out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    analyze()
