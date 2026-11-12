"""Aggregate paired runs by simulated world, without treating optimizer seeds as independent worlds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize_results(out,settings):
    out=Path(out);rows=[];repeat_count=0;convergence=[]
    for path in sorted((out/'runs').glob('*.json')):
        result=json.loads(path.read_text());rows.extend(result['metrics']);repeat_count+=int(result['exact_repeat_checked'])
        r=result['metrics'][0]
        history=pd.read_csv(path.with_name(path.stem+'_history.csv'))
        best=history[history.epoch.eq(r['best_epoch'])].iloc[0]
        convergence.append({key:r[key] for key in ['machine','simulation_seed','seed','mechanism','completeness','method','best_epoch','epochs_run']})
        convergence[-1].update(train_observed_mae_at_best=float(best.train_mae_hours),
                              validation_mae_at_best=float(best.validation_mae_hours),
                              validation_mae_at_end=float(history.iloc[-1].validation_mae_hours),
                              epoch_budget_reached=bool(r['epochs_run']==settings['max_epochs']))
    m=pd.DataFrame(rows)
    expected=len(settings['simulation_seeds'])*3*len(settings['paired_seeds'])*(len(settings['label_completeness'])+len(settings['quantity_completeness'])-1)*2*2
    if len(m)!=expected:raise ValueError(f'Incomplete sweep: {len(m)} metric rows, expected {expected}')
    # Identical complete-input condition: reuse its fits in the second mechanism's display only.
    control=m[m.completeness.eq(1.)].copy();control['mechanism']='quantity'
    m=pd.concat([m,control],ignore_index=True)
    numeric=['MAE','RMSE','R2','WAPE','constraint_violation','mean_accounting_excess_hours']
    if not np.isfinite(m[numeric]).all().all():raise ValueError('Nonfinite metric')
    m.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(convergence).to_csv(out/'convergence_diagnostics.csv',index=False)
    world=m.groupby(['mechanism','machine','target','completeness','method','simulation_seed'])[numeric].mean().reset_index()
    world.to_csv(out/'world_metrics.csv',index=False)
    aggregate=world.groupby(['mechanism','machine','target','completeness','method']).agg(
        MAE_mean=('MAE','mean'),MAE_world_sd=('MAE','std'),RMSE_mean=('RMSE','mean'),R2_mean=('R2','mean'),
        violation_mean=('constraint_violation','mean'),violation_world_sd=('constraint_violation','std')).reset_index()
    aggregate.to_csv(out/'metrics_aggregated.csv',index=False)
    paired=m.pivot(index=['mechanism','machine','target','completeness','simulation_seed','seed'],columns='method',values='MAE').reset_index()
    paired['relative_improvement']=(paired.data_driven-paired.domain_informed)/paired.data_driven
    paired.to_csv(out/'paired_comparison.csv',index=False)
    effect_world=paired.groupby(['mechanism','machine','target','completeness','simulation_seed']).relative_improvement.mean().reset_index()
    effect_world.to_csv(out/'paired_world_effects.csv',index=False)
    effects=effect_world.groupby(['mechanism','machine','target','completeness']).agg(
        improvement_mean=('relative_improvement','mean'),improvement_world_sd=('relative_improvement','std'),
        minimum_world_improvement=('relative_improvement','min'),maximum_world_improvement=('relative_improvement','max')).reset_index()
    effects.to_csv(out/'effect_summary.csv',index=False)
    macro_world=world.groupby(['mechanism','target','completeness','method','simulation_seed'])[numeric].mean().reset_index()
    macro=macro_world.groupby(['mechanism','target','completeness','method']).agg(MAE_mean=('MAE','mean'),MAE_world_sd=('MAE','std'),
                          violation_mean=('constraint_violation','mean'),violation_world_sd=('constraint_violation','std')).reset_index()
    macro.to_csv(out/'macro_metrics.csv',index=False)
    macro_effect_world=effect_world.groupby(['mechanism','target','completeness','simulation_seed']).relative_improvement.mean().reset_index()
    macro_effect=macro_effect_world.groupby(['mechanism','target','completeness']).agg(improvement_mean=('relative_improvement','mean'),
                               improvement_world_sd=('relative_improvement','std'),minimum_world_improvement=('relative_improvement','min')).reset_index()
    macro_effect.to_csv(out/'macro_effects.csv',index=False)
    aggregate[aggregate.target.eq('elapsed_hours')].to_latex(out/'paper_table.tex',index=False,escape=True,float_format='%.3f',
                 caption='Synthetic degradation. Standard deviations across three simulation worlds after averaging paired mask/optimizer seeds.')
    figures(out,m,aggregate,macro,macro_effect)
    lines=['# Synthetic observation degradation results','',
        'This experiment tests the previously proposed soft accounting constraints on fresh synthetic worlds. '
        'It does not alter the industrial data or run industrial MS3.','',
        '## Frozen protocol','',
        'Three fresh simulation seeds (811, 977, 1231), three machines and two paired mask/optimizer seeds '
        '(101, 202) per world. Each machine has 360 training, 120 validation and 120 held-out test jobs. '
        'Architectures and positive constraint weights were frozen from the earlier validation-selected synthetic run '
        '(world 731), never tuned on these test samples. Both methods share inputs, masks, initialization, batch order '
        'and a 160-epoch / 25-patience budget. Stopping epochs may differ. There are 288 fits plus six exact-repeat checks.','',
        '**Label removal (main experiment):** independently mask elapsed and booked training labels at '
        '100%, 75%, 50%, 25%, 10%, 5%. Exactly floor(n × fraction) observations per target are retained. '
        'Masks are nested across levels and independent of target values. All 360 feature rows remain available; '
        'constraints may operate on unlabeled rows. This is label sparsity, not wholesale row deletion. '
        'Train target scales use retained labels only. Hidden values are replaced by NaN before scale fitting and '
        'by masked placeholders in tensors. Constraints receive predictions and no target data.','',
        '**Complete validation anchor:** all 120 validation jobs retain both labels for early stopping. '
        'At 5%, the learner still receives 18 labels per target in training plus 120 per target in validation. '
        'Therefore this experiment measures sparse training supervision with complete calibration, not 5% of all '
        'development observations. This anchor is shared by both methods and can substantially moderate degradation.','',
        '**Quantity removal (separate experiment):** retain quantity for 100%, 25%, 5% of jobs in training, '
        'validation and test; duration labels stay complete. Replace missing quantities with the median of observed '
        'training quantities and include a missingness flag. Inputs at test time are degraded but evaluation labels '
        'remain complete. This changes information available at deployment and must not be conflated with training-label '
        'removal. Both mechanisms use the same complete control fit.','',
        '## Macro elapsed-time results','',
        'MAE is in hours. Means weight each machine and simulation world equally. SD is across three worlds after '
        'averaging the two mask/optimizer repeats within each world. It is descriptive, not a confidence interval.','',
        '| Mechanism | Available | Data-driven MAE | Domain-informed MAE | Paired improvement |',
        '|---|---:|---:|---:|---:|']
    for mechanism in ['labels','quantity']:
        for level in sorted(m[m.mechanism.eq(mechanism)].completeness.unique(),reverse=True):
            z=macro[(macro.mechanism==mechanism)&(macro.target=='elapsed_hours')&(macro.completeness==level)].set_index('method')
            e=macro_effect[(macro_effect.mechanism==mechanism)&(macro_effect.target=='elapsed_hours')&(macro_effect.completeness==level)].iloc[0]
            lines.append(f'| {mechanism} | {level:.0%} | {z.loc["data_driven","MAE_mean"]:.3f} ± {z.loc["data_driven","MAE_world_sd"]:.3f} | '
                         f'{z.loc["domain_informed","MAE_mean"]:.3f} ± {z.loc["domain_informed","MAE_world_sd"]:.3f} | {e.improvement_mean:+.1%} |')
    lines+=['','## Interpretation','',
       'Positive improvement means lower constrained MAE; negative means worse. A descriptive material-effect threshold '
       'of 5% was configured before the run. It is not a statistical significance test.']
    for mechanism in ['labels','quantity']:
        e=effects[(effects.mechanism==mechanism)&(effects.target=='elapsed_hours')&(effects.completeness<1)]
        count=((e.improvement_mean>=settings['material_improvement_threshold'])&(e.minimum_world_improvement>0)).sum()
        lines.append(f'- {mechanism}: {int(count)} of {len(e)} machine/level combinations have at least 5% mean paired improvement '
                     'and positive mean effects in all three worlds.')
        for machine in sorted(e.machine.unique()):
            g=e[e.machine==machine]
            best=g.loc[g.improvement_mean.idxmax()];worst=g.loc[g.improvement_mean.idxmin()]
            lines.append(f'  - {machine}: best {best.improvement_mean:+.1%} at {best.completeness:.0%}; '
                         f'worst {worst.improvement_mean:+.1%} at {worst.completeness:.0%}.')
    lines+=['','The constraints do not encode quantity/rate equations. They only enforce nonnegativity and booked time '
        'within elapsed time. They cannot supply an unobserved workload. Constraint violations, per-target errors, '
        'per-world paired effects and negative effects are all retained in the exported tables. No constraint ablation '
        'was performed, so individual causal contributions cannot be claimed.','',
        'Limitations: one simple stationary generator family; only three independent simulated worlds; complete '
        'validation anchor for label removal; frozen hyperparameters rather than per-level optimization; fixed '
        'optimization budgets; simultaneous input loss at deployment in the quantity condition. '
        'No test-driven retuning was performed. Seed variation is not evidence of statistical significance.','',
        '## Validation and reproducibility','',
        f'{repeat_count} low-completeness fits were retrained exactly with matching parameters and histories. '
        'Every test prediction and reported metric is finite. Masks and fitted preprocessing are stored per condition. '
        'Run JSON files are written last, allowing interrupted runs to resume only with identical core code/configuration. '
        'Prior industrial and synthetic outputs are hash-protected.','',
        '```powershell','$env:PYTHONPATH = "$PWD/src"','python -m pytest -q --basetemp=.pytest_tmp',
        'python -m linn.experiments.synthetic_degradation --config configs/synthetic_degradation.json','```','',
        'Outputs: metrics.csv, metrics_aggregated.csv, macro_metrics.csv, paired_comparison.csv, paired_world_effects.csv, '
        'effect_summary.csv, protocol.json, environment.json, conditions/ (masks and scales), references/ (synthetic worlds), '
        'runs/ (predictions and loss histories), figures/ (PNG/PDF), and paper_table.tex. '
        'Use degradation_overview, per_machine_labels and reconstruction_examples for the paper; report the validation '
        'anchor explicitly. Industrial MS3 remains unstarted.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(macro[macro.target.eq('elapsed_hours')].to_string(index=False),flush=True)


def figures(out,m,aggregate,macro,effects):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    colors={'data_driven':'#0072B2','domain_informed':'#D55E00'}
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(folder/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for method,color in colors.items():
        for mechanism,ax in [('labels',axs[0,0]),('quantity',axs[1,1])]:
            g=macro[(macro.method==method)&(macro.mechanism==mechanism)&(macro.target=='elapsed_hours')].sort_values('completeness')
            x=g.completeness.to_numpy()*100;y=g.MAE_mean.to_numpy();sd=g.MAE_world_sd.to_numpy()
            ax.plot(x,y,'o-',label=method,color=color);ax.fill_between(x,y-sd,y+sd,color=color,alpha=.15)
        g=macro[(macro.method==method)&(macro.mechanism=='labels')&(macro.target=='elapsed_hours')].sort_values('completeness')
        axs[0,1].plot(g.completeness.to_numpy()*100,g.violation_mean.to_numpy()*100,'o-',label=method,color=color)
    g=effects[(effects.mechanism=='labels')&(effects.target=='elapsed_hours')].sort_values('completeness')
    axs[1,0].errorbar(g.completeness.to_numpy()*100,g.improvement_mean.to_numpy()*100,yerr=g.improvement_world_sd.to_numpy()*100,fmt='o-',color='#009E73')
    axs[1,0].axhline(0,color='black',lw=1)
    axs[0,0].set(title='Training duration labels removed',ylabel='Elapsed MAE (hours)')
    axs[0,1].set(title='Label removal: physical accounting',ylabel='Any violation (%)')
    axs[1,0].set(title='Label removal: paired effect',ylabel='Relative MAE improvement (%)')
    axs[1,1].set(title='Quantity removed in train/validation/test',ylabel='Elapsed MAE (hours)')
    for ax in axs.flat:ax.set_xlabel('Training labels available (%)');ax.invert_xaxis()
    axs[1,1].set_xlabel('Quantity inputs available (%)')
    for ax in [axs[0,0],axs[0,1],axs[1,1]]:ax.legend(fontsize=8)
    save(fig,'degradation_overview')
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,machine in zip(axs,sorted(m.machine.unique())):
        for method,color in colors.items():
            g=aggregate[(aggregate.machine==machine)&(aggregate.method==method)&(aggregate.mechanism=='labels')&(aggregate.target=='elapsed_hours')].sort_values('completeness')
            ax.errorbar(g.completeness.to_numpy()*100,g.MAE_mean.to_numpy(),yerr=g.MAE_world_sd.to_numpy(),fmt='o-',color=color,label=method)
        ax.set(title=machine,xlabel='Training labels available (%)',ylabel='Elapsed MAE (h)');ax.invert_xaxis();ax.legend(fontsize=8)
    save(fig,'per_machine_labels')
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    world=int(m.simulation_seed.min());seed=int(m.seed.min());machine=sorted(m.machine.unique())[0]
    for ax,level in zip(axs,[100,25,5]):
        for method,color in colors.items():
            p=pd.read_csv(out/'runs'/f'{world}_{machine}_{seed}_labels_{level:03d}_{method}_predictions.csv').iloc[:35]
            ax.plot(np.arange(len(p)),p.elapsed_predicted.to_numpy(),label=method,color=color)
        ax.plot(np.arange(len(p)),p.elapsed_observed.to_numpy(),'k--',label='observed')
        ax.set(title=f'{level}% labels',xlabel='First held-out jobs',ylabel='Elapsed hours');ax.legend(fontsize=7)
    fig.suptitle(f'Illustration only: {machine}, world {world}, seed {seed}')
    save(fig,'reconstruction_examples')
