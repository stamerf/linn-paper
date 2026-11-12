import json
from pathlib import Path
import numpy as np
import pytest
import torch
import pandas as pd
from linn.data.simulation import simulate_workstations
from linn.experiments.synthetic_confirmation import prepare_sparse
from linn.evaluation.confirmation_report import paired_interval, summarize
from linn.training.tuning import train_candidate
from linn.evaluation.metrics import regression_metrics, workstation_violations


def sample():
    cfg = json.loads(Path('configs/workstation_synthetic.json').read_text())['simulation']
    return simulate_workstations(dict(cfg, seed=55501)).query("machine == 'SIM_A'")


def test_sparse_validation_and_hidden_values_cannot_influence_fit():
    frame = sample()
    ds, spec, audit, meta = prepare_sparse(frame, .05, 7701)
    assert ds['train'].mask.sum(0).tolist() == [18, 18]
    assert ds['validation'].mask.sum(0).tolist() == [6, 6]
    assert ds['test'].mask.sum(0).tolist() == [120, 120]
    altered = frame.copy()
    for target, flag in [('elapsed_hours', 'elapsed_observed'), ('booked_hours', 'booked_observed')]:
        hidden = audit.loc[~audit[flag], 'record_id']
        altered.loc[altered.record_id.isin(hidden), target] = 1e9
    other, spec2, _, meta2 = prepare_sparse(altered, .05, 7701)
    assert meta == meta2 and spec.context == spec2.context
    for split in ds:
        assert torch.equal(ds[split].x, other[split].x)
        assert torch.equal(ds[split].y, other[split].y)
    candidate = {'hidden_dims': [8], 'learning_rate': .003, 'weight_decay': .0001, 'batch_size': 32}
    settings = {'max_epochs': 2, 'patience': 2, 'min_delta': .0001}
    model, history, info = train_candidate(ds['train'], ds['validation'], spec, candidate, .1, 42, settings)
    again, h2, i2 = train_candidate(other['train'], other['validation'], spec2, candidate, .1, 42, settings)
    assert history == h2 and info == i2
    assert all(torch.equal(v, again.model.state_dict()[k]) for k, v in model.model.state_dict().items())


def test_test_targets_cannot_change_development_data():
    frame = sample()
    ds, spec, _, meta = prepare_sparse(frame, .05, 7701)
    frame.loc[frame.split.eq('test'), ['elapsed_hours', 'booked_hours']] = 9999.
    other, spec2, _, meta2 = prepare_sparse(frame, .05, 7701)
    assert meta == meta2 and spec.context == spec2.context
    for split in ['train', 'validation']:
        assert torch.equal(ds[split].y, other[split].y)
        assert torch.equal(ds[split].x, other[split].x)


def test_world_bootstrap_is_deterministic_and_directional():
    values = np.linspace(.1, .3, 20)
    result = paired_interval(values)
    assert result == paired_interval(values)
    assert result['mean'] == pytest.approx(.2)
    assert .1 < result['ci_low'] < result['mean'] < result['ci_high'] < .3
    negative = paired_interval(-values)
    assert negative['ci_high'] < 0
    constant = paired_interval([.1]*20)
    assert constant['ci_low'] == pytest.approx(.1)
    assert constant['ci_high'] == pytest.approx(.1)
    with pytest.raises(ValueError):
        paired_interval([1.])


def test_confirmation_seeds_are_fresh():
    config = json.loads(Path('configs/synthetic_confirmation.json').read_text())
    seeds = config['simulation_seeds']
    assert len(seeds) == len(set(seeds)) == 20
    assert not set(seeds) & {731, 811, 977, 1231}


def test_report_averages_paired_effects_not_ratio_of_pooled_errors(tmp_path):
    # Deliberately unequal baseline errors distinguish the two estimands.
    settings = json.loads(Path('configs/synthetic_confirmation.json').read_text())
    runs = tmp_path/'runs'
    runs.mkdir()
    for world in settings['simulation_seeds']:
        for i, machine in enumerate(['SIM_A', 'SIM_B', 'SIM_C']):
            for level in settings['label_completeness']:
                for method in ['data_driven', 'domain_informed']:
                    error = (2.**i)*(1 if method == 'data_driven' else 1-.1*(i+1))
                    rows = [{'world': world, 'machine': machine, 'completeness': level,
                             'target': target, 'method': method, 'MAE': error,
                             'RMSE': error, 'R2': .5, 'constraint_violation': 0.,
                             'epoch_budget_reached': False}
                            for target in ['elapsed_hours', 'booked_hours']]
                    (runs/f'{world}_{machine}_{level}_{method}.json').write_text(
                        json.dumps({'metrics': rows, 'exact_repeat_checked': False}))
    summarize(tmp_path, settings)
    primary = json.loads((tmp_path/'primary_result.json').read_text())
    assert primary['mean'] == pytest.approx(.2)
    assert primary['ci_low'] == pytest.approx(.2)
    assert primary['ci_high'] == pytest.approx(.2)
    assert primary['n_worlds'] == 20
    assert (tmp_path/'figures/confirmation.pdf').is_file()


@pytest.mark.skipif(not Path('results/synthetic_confirmation/primary_result.json').exists(),
                    reason='Complete confirmation artifacts not yet available')
def test_completed_confirmation_predictions_masks_and_primary_interval():
    out = Path('results/synthetic_confirmation')
    settings = json.loads(Path('configs/synthetic_confirmation.json').read_text())
    files = sorted((out/'runs').glob('*.json'))
    assert len(files) == 240
    repeats = 0
    for path in files:
        result = json.loads(path.read_text())
        repeats += int(result['exact_repeat_checked'])
        predictions = pd.read_csv(path.with_name(path.stem+'_predictions.csv'))
        p = predictions[['elapsed_predicted', 'booked_predicted']].to_numpy()
        for j, row in enumerate(result['metrics']):
            y = predictions[['elapsed_observed', 'booked_observed']].to_numpy()[:, j]
            for metric, value in regression_metrics(y, p[:, j]).items():
                assert row[metric] == pytest.approx(value, rel=1e-6, abs=1e-6)
            assert row['constraint_violation'] == pytest.approx(workstation_violations(p, 0.)['any_violation_fraction'])
            for target in ['elapsed_hours', 'booked_hours']:
                assert row['n_train_'+target] == int(360*row['completeness'])
                assert row['n_validation_'+target] == int(120*row['completeness'])
    assert repeats == 2
    for path in (out/'conditions').glob('*_mask.csv'):
        audit = pd.read_csv(path)
        level = int(path.stem.split('_')[-2])/100
        assert not audit.record_id.duplicated().any()
        for split, n in [('train', int(360*level)), ('validation', int(120*level)), ('test', 120)]:
            assert audit.loc[audit.split.eq(split), ['elapsed_observed', 'booked_observed']].sum().tolist() == [n, n]
    metrics = pd.read_csv(out/'metrics.csv')
    m = metrics[(metrics.target == 'elapsed_hours') & (metrics.completeness == .05)]
    p = m.pivot(index=['world', 'machine'], columns='method', values='MAE')
    world_effect = ((p.data_driven-p.domain_informed)/p.data_driven).groupby('world').mean()
    assert set(world_effect.index) == set(settings['simulation_seeds'])
    expected = paired_interval(world_effect.to_numpy(), settings['bootstrap_seed'], settings['bootstrap_replicates'])
    saved = json.loads((out/'primary_result.json').read_text())
    for key, value in expected.items():
        assert saved[key] == pytest.approx(value)
