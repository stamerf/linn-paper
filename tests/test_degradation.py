import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from linn.evaluation.degradation import observation_mask,ObservedMedianImputer
from linn.experiments.synthetic_degradation import prepare_condition
from linn.data.simulation import simulate_workstations
from linn.training.tuning import train_candidate,predict
from linn.evaluation.metrics import regression_metrics,workstation_violations


def frame():
    cfg=json.loads(Path('configs/workstation_synthetic.json').read_text())['simulation']
    cfg['jobs_per_machine']=100
    f=simulate_workstations(cfg)
    return f[f.machine.eq('SIM_A')]


def test_masks_exact_nested_paired_and_row_order_invariant():
    ids=np.array([f'job{i}' for i in range(100)])
    low=observation_mask(ids,.1,42,2);high=observation_mask(ids,.5,42,2)
    assert low.sum(axis=0).tolist()==[10,10]
    assert np.all(~low|high)
    np.testing.assert_array_equal(low,observation_mask(ids,.1,42,2))
    np.testing.assert_array_equal(low[::-1],observation_mask(ids[::-1],.1,42,2))
    assert not np.array_equal(low,observation_mask(ids,.1,43,2))
    assert observation_mask(ids,1,1,2).all() and not observation_mask(ids,0,1,2).any()
    whole=observation_mask(ids,.5,42,2,whole_record=True)
    np.testing.assert_array_equal(whole[:,0],whole[:,1])
    assert not np.array_equal(high[:,0],high[:,1])


@pytest.mark.parametrize('fraction',[-.1,1.1])
def test_invalid_completeness_rejected(fraction):
    with pytest.raises(ValueError):observation_mask(['a'],fraction,1)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):observation_mask(['a','a'],.5,1)


def test_hidden_labels_cannot_change_scale_features_or_constraint_inputs():
    f=frame();ds,spec,audit,meta=prepare_condition(f,'labels',.1,42)
    changed=f.copy()
    for target,flag in [('elapsed_hours','elapsed_observed'),('booked_hours','booked_observed')]:
        hidden=set(audit.loc[~audit[flag],'record_id'])
        changed.loc[changed.record_id.isin(hidden),target]=1e9
    other,spec2,audit2,meta2=prepare_condition(changed,'labels',.1,42)
    assert spec.context==spec2.context and meta==meta2
    pd.testing.assert_frame_equal(audit,audit2)
    for phase in ds:
        np.testing.assert_array_equal(ds[phase].x.numpy(),other[phase].x.numpy())
        np.testing.assert_array_equal(ds[phase].y.numpy(),other[phase].y.numpy())
    assert ds['train'].mask.sum().item()==12
    assert ds['validation'].mask.all() and ds['test'].mask.all()
    assert ds['train'][0]['constraint_inputs']=={}


def test_hidden_quantity_cannot_change_imputation_or_scaled_features():
    f=frame();ds,spec,audit,meta=prepare_condition(f,'quantity',.25,42)
    changed=f.copy();hidden=set(audit.loc[~audit.quantity_observed,'record_id'])
    changed.loc[changed.record_id.isin(hidden),'planned_quantity']=1e9
    other,spec2,_,meta2=prepare_condition(changed,'quantity',.25,42)
    assert meta==meta2 and spec.context==spec2.context
    for phase in ds:
        np.testing.assert_array_equal(ds[phase].x.numpy(),other[phase].x.numpy())
        assert ds[phase].mask.all()
    with pytest.raises(ValueError):ObservedMedianImputer().fit([np.nan],[False])


def test_masked_training_diagnostics_use_observed_values_only():
    ds,spec,_,_=prepare_condition(frame(),'labels',.1,42)
    settings={'max_epochs':2,'patience':2,'min_delta':.0001}
    candidate={'hidden_dims':[8],'learning_rate':.003,'weight_decay':.0001,'batch_size':16}
    trainer,history,info=train_candidate(ds['train'],ds['validation'],spec,candidate,.1,42,settings)
    # Restored checkpoint agrees with the logged training MAE computed on observations only.
    errors=predict(trainer,ds['train'])-ds['train'].y.numpy()
    actual=np.abs(errors[ds['train'].mask.numpy()]).mean()
    assert actual==pytest.approx(history[info['best_epoch']-1]['train_mae_hours'])
    assert np.isfinite(pd.DataFrame(history).to_numpy()).all()


@pytest.mark.skipif(not Path('results/synthetic_degradation/metrics.csv').exists(),reason='Completed degradation sweep absent')
def test_saved_sweep_is_complete_paired_and_recomputable():
    out=Path('results/synthetic_degradation')
    files=sorted((out/'runs').glob('*.json'))
    assert len(files)==288
    repeats=0
    for path in files:
        result=json.loads(path.read_text());repeats+=int(result['exact_repeat_checked'])
        pred=pd.read_csv(path.with_name(path.stem+'_predictions.csv'))
        p=pred[['elapsed_predicted','booked_predicted']].to_numpy()
        for j,r in enumerate(result['metrics']):
            y=pred[['elapsed_observed','booked_observed']].to_numpy()[:,j]
            for metric,value in regression_metrics(y,p[:,j]).items():
                # CSV serializes float32 predictions to decimal; allow sub-micro
                # differences when reloading as float64, including near-zero bias.
                assert r[metric]==pytest.approx(value, rel=1e-6, abs=1e-6)
            assert r['constraint_violation']==pytest.approx(workstation_violations(p,0.)['any_violation_fraction'])
        r=result['metrics'][0]
        if r['mechanism']=='labels':
            assert r['n_observed_elapsed']==int(360*r['completeness'])
            assert r['n_observed_booked']==int(360*r['completeness'])
        else:
            assert r['n_observed_elapsed']==r['n_observed_booked']==360
    assert repeats==6
    metrics=pd.read_csv(out/'metrics.csv')
    assert len(metrics)==648
    assert metrics.groupby(['simulation_seed','machine','seed','mechanism','completeness','target']).method.nunique().eq(2).all()
