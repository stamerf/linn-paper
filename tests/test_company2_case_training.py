import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from linn.experiments.company2_case import select_case, prepare_development, baseline_predictions
from linn.training.tuning import train_candidate, predict
from linn.evaluation.metrics import regression_metrics, workstation_violations


def fixture_frame():
    n = 80
    return pd.DataFrame({'record_id': [f'job{i:03}' for i in range(n)],
                         'order_id': [str(i) for i in range(n)], 'machine': 'PRESS1',
                         'product': ['P1', 'P2']*(n//2), 'operation': '10',
                         'split': ['train']*40+['validation']*20+['test']*20,
                         'confirmed_quantity': np.arange(1, n+1, dtype=float),
                         'elapsed_hours': np.arange(1, n+1)*.1+1,
                         'booked_hours': np.arange(1, n+1)*.1+.9})


def test_selection_cannot_use_test_labels_or_volume():
    f = fixture_frame()
    _, selected, _ = select_case(f, {'min_train': 15, 'min_validation': 5})
    _, without_test, _ = select_case(f[f.split.ne('test')], {'min_train': 15, 'min_validation': 5})
    assert selected == without_test == ['PRESS1']


def test_hidden_labels_and_test_data_cannot_affect_preprocessing_or_training():
    f = fixture_frame()
    ds, encoder, spec, masks = prepare_development(f, .25, 91)
    assert ds['train'].mask.sum(0).tolist() == [10, 10]
    assert ds['validation'].mask.sum(0).tolist() == [5, 5]
    altered = f.copy()
    for name, flag in [('elapsed_hours', 'elapsed_observed'), ('booked_hours', 'booked_observed')]:
        altered.loc[altered.record_id.isin(masks.loc[~masks[flag], 'record_id']), name] = 1e9
        altered.loc[altered.split.eq('test'), name] = 1e9
    altered.loc[altered.split.eq('test'), 'confirmed_quantity'] = 1e10
    ds2, enc2, spec2, _ = prepare_development(altered, .25, 91)
    assert encoder.metadata() == enc2.metadata() and spec.context == spec2.context
    for split in ds:
        assert torch.equal(ds[split].x, ds2[split].x)
        assert torch.equal(ds[split].y, ds2[split].y)
    candidate = {'hidden_dims': [8], 'learning_rate': .003, 'weight_decay': .001, 'batch_size': 16}
    cfg = {'max_epochs': 2, 'patience': 2, 'min_delta': .0001}
    trainer, history, info = train_candidate(ds['train'], ds['validation'], spec, candidate, .1, 17, cfg)
    again, h2, i2 = train_candidate(ds2['train'], ds2['validation'], spec2, candidate, .1, 17, cfg)
    assert history == h2 and info == i2
    np.testing.assert_array_equal(predict(trainer, ds['validation']), predict(again, ds2['validation']))
    for method in ['median', 'quantity_linear']:
        np.testing.assert_array_equal(baseline_predictions(ds['train'], ds['validation'], method),
                                      baseline_predictions(ds2['train'], ds2['validation'], method))


@pytest.mark.skipif(not Path('results/company2_case/metrics.csv').exists(), reason='Industrial case has not completed')
def test_completed_case_artifacts():
    out = Path('results/company2_case')
    settings = json.loads(Path('configs/company2_case.json').read_text())
    metrics = pd.read_csv(out/'metrics.csv')
    assert len(metrics) == 260  # 65 method runs, two splits, two targets.
    assert metrics.method.nunique() == 5
    repeats = 0
    for path in (out/'runs').glob('*.json'):
        result = json.loads(path.read_text()); repeats += int(result['exact_repeat_checked'])
        p = pd.read_csv(path.with_name(path.stem+'_predictions.csv'))
        for r in result['metrics']:
            split = p[p.split.eq(r['split'])]
            key = 'elapsed' if r['target'] == 'elapsed_hours' else 'booked'
            observed = split[key+'_observed'].notna()
            actual = regression_metrics(split.loc[observed, key+'_observed'], split.loc[observed, key+'_predicted'])
            for name, value in actual.items():
                assert r[name] == pytest.approx(value, abs=2e-6, rel=1e-6)
            violations = workstation_violations(split[['elapsed_predicted', 'booked_predicted']].to_numpy(), .02)
            assert r['constraint_violation'] == pytest.approx(violations['any_violation_fraction'])
            assert r['n_observed_train_per_target'] == int(147*r['completeness'])
            assert r['n_observed_validation_per_target'] == int(47*r['completeness'])
    assert repeats == 2
    frame = pd.read_csv(out/'prepared_operations.csv')
    assert frame.groupby('order_id').split.nunique().max() == 1
    assert sorted(frame.machine.unique()) == ['PSK 25', 'PSK 26', 'PSK 27', 'PSK 28', 'PSK 29']
    history = pd.read_csv(out/'history.csv')
    assert np.isfinite(history.select_dtypes(include=np.number).to_numpy()).all()
