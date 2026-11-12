"""World-clustered paired analysis; never resample jobs as independent experiments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from linn.experiments.workstation_ms1 import save_json


def paired_interval(values, seed=61031, repetitions=20000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Need at least two finite independent world effects')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return {'mean': float(values.mean()), 'ci_low': float(low), 'ci_high': float(high),
            'world_sd': float(values.std(ddof=1)), 'n_worlds': len(values)}


def summarize(out, settings):
    out = Path(out)
    results = [json.loads(p.read_text()) for p in sorted((out/'runs').glob('*.json'))]
    m = pd.DataFrame([row for r in results for row in r['metrics']])
    assert len(results) == len(settings['simulation_seeds'])*3*len(settings['label_completeness'])*2
    assert np.isfinite(m[['MAE', 'RMSE', 'R2', 'constraint_violation']].to_numpy()).all()
    keys = ['world', 'machine', 'completeness', 'target']
    assert not m.duplicated(keys+['method']).any()
    assert m.groupby(keys).method.nunique().eq(2).all()
    assert set(m.world) == set(settings['simulation_seeds'])
    assert m.groupby(['world', 'completeness', 'target', 'method']).machine.nunique().eq(3).all()
    m.to_csv(out/'metrics.csv', index=False)
    paired = m.pivot(index=keys, columns='method', values='MAE').reset_index()
    paired['absolute_improvement_hours'] = paired.data_driven-paired.domain_informed
    paired['relative_improvement'] = paired.absolute_improvement_hours/paired.data_driven
    paired.to_csv(out/'paired_machine_effects.csv', index=False)
    world = paired.groupby(['world', 'completeness', 'target'])[
        ['data_driven', 'domain_informed', 'absolute_improvement_hours', 'relative_improvement']].mean().reset_index()
    world.to_csv(out/'world_effects.csv', index=False)
    intervals = []
    for (level, target), group in world.groupby(['completeness', 'target']):
        for metric in ['absolute_improvement_hours', 'relative_improvement']:
            intervals.append({'completeness': level, 'target': target, 'metric': metric,
                              **paired_interval(group[metric], settings['bootstrap_seed'], settings['bootstrap_replicates'])})
    intervals = pd.DataFrame(intervals)
    intervals.to_csv(out/'paired_intervals.csv', index=False)
    per_machine = []
    for (machine, level, target), group in paired.groupby(['machine', 'completeness', 'target']):
        per_machine.append({'machine': machine, 'completeness': level, 'target': target,
                            **paired_interval(group.relative_improvement, settings['bootstrap_seed'], settings['bootstrap_replicates'])})
    pd.DataFrame(per_machine).to_csv(out/'machine_intervals_descriptive.csv', index=False)
    table = m.groupby(['completeness', 'target', 'method']).agg(
        MAE=('MAE', 'mean'), RMSE=('RMSE', 'mean'), R2=('R2', 'mean'),
        constraint_violation=('constraint_violation', 'mean'), budget_hits=('epoch_budget_reached', 'sum')).reset_index()
    table.to_csv(out/'aggregate_metrics.csv', index=False)
    table.to_latex(out/'paper_table.tex', index=False, float_format='%.4f',
                   caption='Sparse training and validation labels: macro metrics over 20 synthetic worlds and three machines.')
    primary = intervals[(intervals.completeness == settings['primary_completeness']) &
                        (intervals.target == 'elapsed_hours') & (intervals.metric == 'relative_improvement')].iloc[0].to_dict()
    primary['supports_positive_effect'] = bool(primary['ci_low'] > 0)
    primary['supports_practical_threshold'] = bool(primary['ci_low'] > settings['practical_threshold'])
    save_json(out/'primary_result.json', primary)
    figures = out/'figures'; figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    g = world[(world.target == 'elapsed_hours') & (world.completeness == settings['primary_completeness'])].sort_values('world')
    diagnostics = {'positive_worlds': int((g.relative_improvement > 0).sum()),
                   'negative_worlds': int((g.relative_improvement < 0).sum()),
                   'median_world_relative_improvement': float(g.relative_improvement.median()),
                   'minimum_world_relative_improvement': float(g.relative_improvement.min()),
                   'maximum_world_relative_improvement': float(g.relative_improvement.max())}
    save_json(out/'effect_diagnostics.json', diagnostics)
    axes[0].plot(np.arange(len(g)), g.data_driven.to_numpy(), 'o-', label='Data-driven')
    axes[0].plot(np.arange(len(g)), g.domain_informed.to_numpy(), 'o-', label='Domain-informed')
    axes[0].set(xlabel='Independent world index', ylabel='Macro elapsed MAE (h)', title='5% training and validation labels')
    axes[0].legend()
    axes[1].scatter(g.relative_improvement.to_numpy()*100, np.arange(len(g)), label='World effects')
    axes[1].axvline(0, color='black', lw=1)
    axes[1].axvline(settings['practical_threshold']*100, color='grey', ls='--', label='5% practical threshold')
    axes[1].axvspan(primary['ci_low']*100, primary['ci_high']*100, color='green', alpha=.15, label='95% CI for mean')
    axes[1].axvline(primary['mean']*100, color='green')
    axes[1].set(xlabel='Paired relative MAE reduction (%)', ylabel='Independent world index', title='World effects and mean uncertainty')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(figures/f'confirmation.{ext}', dpi=180)
    plt.close(fig)
    conclusion = ('The interval supports a positive mean effect.' if primary['supports_positive_effect'] else
                  'The interval does not establish a positive mean effect.')
    practical = ('Its lower bound also exceeds the prespecified 5% practical threshold.' if primary['supports_practical_threshold'] else
                 'Its lower bound does not exceed the prespecified 5% practical threshold.')
    markdown_table = '| ' + ' | '.join(table.columns) + ' |\n'
    markdown_table += '|' + '|'.join(['---']*len(table.columns)) + '|\n'
    markdown_table += '\n'.join('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v)
                                                   for v in row) + ' |'
                                for row in table.itertuples(index=False, name=None))
    lines = ['# Confirmatory synthetic results', '',
             f"Primary mean paired elapsed-MAE reduction: **{primary['mean']:.1%}**, approximate two-sided 95% "
             f"world-bootstrap interval **[{primary['ci_low']:.1%}, {primary['ci_high']:.1%}]**. {conclusion} {practical}", '',
             'The primary outcome averages paired relative reductions across three machines within each world, then '
             'across 20 independent worlds. It is not the ratio of pooled MAEs. Bootstrap resampling preserves each '
             'whole world; jobs and machines are not treated as independent replications.', '',
             '## Macro metrics', '', markdown_table, '',
             '## Scope and interpretation', '',
             f"{diagnostics['positive_worlds']} of 20 worlds have positive macro paired effects; "
             f"the median world effect is {diagnostics['median_world_relative_improvement']:.1%}. "
             f"World effects range from {diagnostics['minimum_world_relative_improvement']:.1%} to "
             f"{diagnostics['maximum_world_relative_improvement']:.1%}.", '',
             'At 5%, only 18 training and 6 validation labels per target are available per machine. Both methods '
             'use identical masks, inputs, initialization and optimization budgets. Test labels remain complete. '
             'Frozen choices originate from an earlier development world; none were selected on these test results. '
             'All 20 worlds are retained, with no optional stopping.', '',
             'Only the elapsed-time effect at 5% is confirmatory. Complete-data control, booked-time, per-machine '
             'intervals and other metrics are descriptive secondary analyses. Percentile bootstrap coverage is '
             'approximate with 20 worlds. This local frozen protocol is not an externally registered study.', '',
             'The generator family and machine parameters remain fixed. Evidence here does not establish industrial '
             'generalization, robustness to other generators, or each constraint\'s contribution. Early stopping '
             'with six labels per target is noisy; all budget hits are exported. Soft constraints do not guarantee '
             'feasibility. No industrial MS3 was run.', '',
             '## Validation and artifacts', '',
             f"{len(results)} fits completed; {sum(r['exact_repeat_checked'] for r in results)} exact-repeat checks. "
             'All metrics are finite. Protocol, source hashes, environment, references, masks, scaling, histories, '
             'predictions, paired effects, confidence intervals, PNG/PDF figure and LaTeX table are saved here.', '',
             '```powershell', '$env:PYTHONPATH = "$PWD/src"', 'python -m pytest -q --basetemp=.pytest_tmp',
             'python -m linn.experiments.synthetic_confirmation --config configs/synthetic_confirmation.json', '```']
    (out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(primary, indent=2), flush=True)
