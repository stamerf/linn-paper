import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from linn.data.simulation import simulate_workstations
from linn.data.workstation_dataset import FeatureEncoder, WorkstationDataset
from linn.processes.workstation import workstation_spec
from linn.training.tuning import train_candidate, predict
from linn.evaluation.metrics import regression_metrics


def settings():
    return json.loads(Path('configs/workstation_synthetic.json').read_text())


def test_generator_reproducibility_and_serial_causality():
    config=settings()['simulation'];config['jobs_per_machine']=30
    f=simulate_workstations(config)
    pd.testing.assert_frame_equal(f,simulate_workstations(config))
    assert f.record_id.is_unique and f.order_id.is_unique
    assert (f.start>=f.arrival).all()
    assert (f.elapsed_hours>=f.booked_hours).all() and (f.booked_hours>0).all()
    np.testing.assert_allclose((f.end-f.start).dt.total_seconds()/3600,f.elapsed_hours,atol=1e-8)
    for _,g in f.groupby('machine'):
        assert (g.start.iloc[1:].to_numpy()>=g.end.iloc[:-1].to_numpy()).all()
        for a,b in [('train','validation'),('validation','test')]:
            assert g[g['split']==a].end.max()<=g[g['split']==b].start.min()
    config['seed']+=1
    assert not f.equals(simulate_workstations(config))


def test_oracle_zero_noise_and_no_target_feature_leakage():
    config=settings()['simulation'];config.update(jobs_per_machine=30,processing_log_noise_sigma=0,pause_random_mean_hours=0)
    f=simulate_workstations(config)
    np.testing.assert_array_equal(f.booked_hours,f.oracle_booked_hours)
    np.testing.assert_array_equal(f.elapsed_hours,f.oracle_elapsed_hours)
    train=f[f['split']=='train'];encoder=FeatureEncoder(['planned_quantity']).fit(train)
    assert encoder.metadata()['numeric_mean']==[train.planned_quantity.mean()]
    changed=f.copy()
    for col in ['elapsed_hours','booked_hours','oracle_elapsed_hours','oracle_booked_hours','truth_queue_wait_hours','truth_pause_hours']:
        changed[col]=99999.
    changed['end']=pd.Timestamp('2099-01-01')
    np.testing.assert_array_equal(encoder.transform(f),encoder.transform(changed))
    changed['planned_quantity']*=2
    assert not np.array_equal(encoder.transform(f),encoder.transform(changed))


@pytest.mark.parametrize('column',['booked_hours','elapsed_hours','end','order_id','split'])
def test_target_or_identifier_cannot_be_numeric_feature(column):
    config=settings()['simulation'];config['jobs_per_machine']=30
    f=simulate_workstations(config)
    with pytest.raises(ValueError,match='Forbidden'):
        FeatureEncoder([column]).fit(f[f['split']=='train'])


def test_synthetic_training_smoke():
    config=settings();config['simulation']['jobs_per_machine']=30
    config.update(max_epochs=3,patience=2)
    f=simulate_workstations(config['simulation']);f=f[f.machine=='SIM_A']
    tr=f[f['split']=='train'];va=f[f['split']=='validation']
    encoder=FeatureEncoder(['planned_quantity']).fit(tr)
    a,b=WorkstationDataset(tr,encoder),WorkstationDataset(va,encoder)
    trainer,history,info=train_candidate(a,b,workstation_spec(tr,0.),config['candidates'][0],.1,1,config)
    assert len(history)<=3 and info['best_epoch']>=1
    assert np.isfinite(predict(trainer,b)).all()


@pytest.mark.skipif(not Path('results/synthetic/predictions.csv').exists(),reason='Synthetic results absent')
def test_saved_results_and_generator_match():
    out=Path('results/synthetic');config=settings()
    generated=simulate_workstations(config['simulation'])
    saved=pd.read_csv(out/'simulated_bookings.csv',parse_dates=['arrival','start','end'])
    for col in ['elapsed_hours','booked_hours','planned_quantity']:
        np.testing.assert_allclose(saved[col],generated[col],rtol=1e-12)
    p=pd.read_csv(out/'predictions.csv');metrics=pd.read_csv(out/'metrics.csv')
    assert np.isfinite(p[['observed','predicted']]).all().all()
    for key,g in p.groupby(['machine','method','seed','split','target']):
        row=metrics
        for col,value in zip(['machine','method','seed','split','target'],key):row=row[row[col]==value]
        assert len(row)==1
        calculated=regression_metrics(g.observed,g.predicted)
        for name in ['MAE','RMSE','R2']:assert row.iloc[0][name]==pytest.approx(calculated[name])
