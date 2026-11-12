"""Retrospective monthly rolling-origin sensitivity, with explicit calibration exposure."""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch

from linn.data.company2_case import prepare_casting_case
from linn.data.loaders import load_company2, sha256
from linn.data.workstation_dataset import WorkstationDataset, TARGETS
from linn.evaluation.metrics import regression_metrics, workstation_violations
from linn.evaluation.projection import project_durations
from linn.experiments.company2_case import prepare_development, baseline_predictions, select_case
from linn.experiments.workstation_ms1 import save_json
from linn.training.tuning import train_candidate, predict


FROZEN = {
    'candidate': {'hidden_dims': [32, 16], 'learning_rate': .003,
                  'weight_decay': .001, 'batch_size': 32},
    'constraint_weight': 1., 'ridge_alpha': 1.,
    'max_epochs': 220, 'patience': 35, 'min_delta': .0001,
    'evaluation_seeds': [101, 202, 303, 404, 505],
    'completeness': [1., .25, .1], 'rounding_tolerance_hours': .02,
    'selection': {'min_train': 15, 'min_validation': 5},
    'test_months': ['2026-04-01', '2026-05-01', '2026-06-01',
                    '2026-07-01', '2026-08-01'],
    'configuration_provenance': 'Fixed original-case settings; later-period complete-label calibration',
    'inferential_status': 'Retrospective sensitivity; not fresh independent industrial confirmation',
}


def monthly_split(frame, test_start, period_start='2026-01-01'):
    """Split complete raw-derived GLOBAL order bounds, including the test upper bound.

    The frame must come from prepare_casting_case; its retained-example starts and
    ends cannot safely reconstruct order bounds. Cross-boundary orders are purged.
    """
    test_start = pd.Timestamp(test_start)
    validation_start = test_start - pd.DateOffset(months=1)
    test_end = test_start + pd.DateOffset(months=1)
    begin = pd.Timestamp(period_start)
    if not begin < validation_start < test_start < test_end:
        raise ValueError('Invalid monthly temporal boundaries')
    required = {'global_order_first', 'global_order_last', 'order_id', 'record_id'}
    if not required <= set(frame):
        raise ValueError('Monthly splitting requires raw-derived global order bounds')
    result = frame.copy()
    # Every operation on an order must carry the same raw-derived order bounds.
    bounds = result.groupby('order_id')[['global_order_first', 'global_order_last']].nunique()
    if bounds.gt(1).any().any():
        raise ValueError('Inconsistent global order bounds')
    first, last = result.global_order_first, result.global_order_last
    result['split'] = 'purged'
    result.loc[first.ge(begin) & last.lt(validation_start), 'split'] = 'train'
    result.loc[first.ge(validation_start) & last.lt(test_start), 'split'] = 'validation'
    result.loc[first.ge(test_start) & last.lt(test_end), 'split'] = 'test'
    assert result.groupby('order_id').split.nunique().max() == 1
    return result, {'period_start': str(begin.date()),
                    'validation_start': str(validation_start.date()),
                    'test_start': str(test_start.date()), 'test_end_exclusive': str(test_end.date())}


def _metrics(predictions, tolerance, keys):
    rows = []
    for values, group in predictions.groupby(keys, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        common = dict(zip(keys, values))
        p = group[['elapsed_predicted', 'booked_predicted']].to_numpy()
        violations = workstation_violations(p, tolerance)['any_violation_fraction']
        for target, channel in zip(TARGETS, ['elapsed', 'booked']):
            rows.append({**common, 'target': target, 'n_test': len(group),
                         'constraint_violation': violations,
                         **regression_metrics(group[channel + '_observed'], group[channel + '_predicted'])})
    return pd.DataFrame(rows)


def summarize(out, fold_audits, original_ids, tolerance):
    predictions = pd.concat([pd.read_csv(p) for p in sorted((out/'private').glob('*_predictions.csv'))],
                            ignore_index=True)
    keys = ['fold', 'completeness', 'seed', 'method']
    per_fold = _metrics(predictions, tolerance, keys)
    per_fold.to_csv(out/'fold_metrics.csv', index=False)
    per_fold.groupby(['fold', 'completeness', 'method', 'target']).agg(
        MAE=('MAE', 'mean'), MAE_seed_sd=('MAE', 'std'), RMSE=('RMSE', 'mean'),
        R2=('R2', 'mean'), constraint_violation=('constraint_violation', 'mean'),
        n_test=('n_test', 'first')).reset_index().to_csv(out/'fold_metrics_aggregated.csv', index=False)
    pooled = _metrics(predictions, tolerance, ['completeness', 'seed', 'method'])
    pooled.to_csv(out/'pooled_metrics_by_seed.csv', index=False)
    aggregate = pooled.groupby(['completeness', 'method', 'target']).agg(
        MAE=('MAE', 'mean'), MAE_seed_sd=('MAE', 'std'), RMSE=('RMSE', 'mean'),
        R2=('R2', 'mean'), constraint_violation=('constraint_violation', 'mean'),
        n_test=('n_test', 'first')).reset_index()
    aggregate.to_csv(out/'pooled_metrics.csv', index=False)
    monthly = per_fold.groupby(['completeness', 'method', 'target']).agg(
        MAE=('MAE', 'mean'), RMSE=('RMSE', 'mean'), R2=('R2', 'mean'),
        constraint_violation=('constraint_violation', 'mean')).reset_index()
    monthly.to_csv(out/'unweighted_month_metrics.csv', index=False)
    paired = pooled[pooled.target.eq('elapsed_hours')].pivot(
        index=['completeness', 'seed'], columns='method', values='MAE').reset_index()
    paired['domain_improvement_vs_data_driven'] = 1 - paired.domain_informed/paired.data_driven
    paired['domain_improvement_vs_nn_projection'] = 1 - paired.domain_informed/paired.data_driven_projection
    paired['domain_improvement_vs_ridge_projection'] = 1 - paired.domain_informed/paired.ridge_projection
    paired.to_csv(out/'paired_effects.csv', index=False)
    ids = set(predictions.record_id)
    unique_rows = predictions[['fold', 'record_id', 'order_id']].drop_duplicates()
    if unique_rows.record_id.duplicated().any():
        raise ValueError('An industrial operation appears in multiple monthly test folds')
    summary = {'status': 'executed', 'completed_folds': int(predictions.fold.nunique()),
               'unique_test_operations': len(ids),
               'unique_test_orders': int(predictions.order_id.nunique()),
               'original_holdout_operations': len(original_ids),
               'overlap_with_original_holdout_operations': len(ids & original_ids),
               'test_operations_additional_to_original_holdout': len(ids - original_ids),
               'seeds_are_independent_observations': False,
               'configuration_calibration_exposure': FROZEN['configuration_provenance'],
               'inferential_status': FROZEN['inferential_status'],
               'folds': fold_audits}
    save_json(out/'summary.json', summary)
    lines = ['# Executed Company 2 monthly temporal sensitivity', '',
             'This is a retrospective sensitivity analysis. Neural hyperparameters were fixed '
             'at the original-case values, which had complete-label calibration on later historical '
             'months. The folds are not independent confirmation or a leakage-free historical '
             'forecasting evaluation. Monthly fitting, feature encoding, scales and stopping use '
             'only each fold\'s past development records.', '',
             f'Completed {summary["completed_folds"]} monthly folds, covering '
             f'{summary["unique_test_operations"]} unique out-of-time operations on '
             f'{summary["unique_test_orders"]} unique production orders. '
             f'{summary["overlap_with_original_holdout_operations"]} overlap the '
             f'{summary["original_holdout_operations"]}-operation original holdout; '
             f'{summary["test_operations_additional_to_original_holdout"]} are additional '
             'historical operations. Five paired fitting seeds do not enlarge these counts.', '',
             '| Test month | Machines | Train | Validation | Test | Test orders | Purged historical operations |',
             '|---|---|---:|---:|---:|---:|---:|']
    for fold in fold_audits:
        if fold['status'] == 'completed':
            lines.append(f'| {fold["fold"]} | {", ".join(fold["selected_machines"])} | '
                         f'{fold["n_train"]} | {fold["n_validation"]} | {fold["n_test"]} | '
                         f'{fold["n_test_orders"]} | {fold["n_purged_historical"]} |')
        else:
            lines.append(f'| {fold["fold"]} | skipped: {fold["reason"]} | | | | | |')
    lines += ['', 'Pooled metrics weight each operation equally within each seed, then average '
              'the five seeds. SD across seeds describes fitting variation, not uncertainty over '
              'independent industrial cohorts. All training and validation target channels are '
              'masked at the stated fraction; test labels remain complete.', '',
              '| Labels | Method | Elapsed MAE (h) | Booked MAE (h) | Elapsed R2 | Joint violations (%) |',
              '|---:|---|---:|---:|---:|---:|']
    for level in FROZEN['completeness']:
        for method in ['ridge', 'ridge_projection', 'data_driven', 'data_driven_projection', 'domain_informed']:
            group = aggregate[(aggregate.completeness == level) & (aggregate.method == method)].set_index('target')
            if group.empty:
                continue
            lines.append(f'| {level:.0%} | {method} | {group.loc["elapsed_hours", "MAE"]:.6f} | '
                         f'{group.loc["booked_hours", "MAE"]:.6f} | '
                         f'{group.loc["elapsed_hours", "R2"]:.6f} | '
                         f'{100*group.loc["elapsed_hours", "constraint_violation"]:.3f} |')
    lines += ['', 'Machine eligibility is determined independently per fold from at least 15 '
              'training and five validation operations. Global raw-derived production-order '
              'bounds determine all split assignments; orders crossing any boundary are purged. '
              'The purged count includes eligible operations on selected machines whose complete '
              'order history begins before the test-window end and crosses a fold boundary.', '',
              'Projection is equal-hour Euclidean projection onto d>=0, b>=0, b<=d+0.02. '
              'It uses no labels or refitting and guarantees zero violations to numerical tolerance. '
              'The fixed Ridge alpha is 1; neural architecture [32,16], learning rate .003, '
              'weight decay .001, batch size 32, domain loss weight 1, at most 220 epochs '
              'with 35-epoch validation patience. The confirmed-quantity feature is retrospective.', '',
              'Run `python tools/company2_temporal_validation.py` from the repository root. '
              'Input/source hashes and the frozen protocol are archived. Source files are '
              'verified unchanged after evaluation. Record-level derivatives, fitted category '
              'identities and masks remain in ignored private outputs.', '']
    (out/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return summary


def run(root):
    root = Path(root).resolve()
    out = root/'results/company2_temporal'
    out.mkdir(parents=True, exist_ok=True)
    (out/'private').mkdir(exist_ok=True)
    torch.set_num_threads(1)
    config_path = root/'configs/workstation.json'
    config = json.loads(config_path.read_text())
    source_paths = [config_path, root/'configs/company2_case.json', Path(__file__),
                    root/'tools/company2_temporal_validation.py']
    for folder in ['data', 'training', 'models', 'constraints', 'processes', 'evaluation']:
        source_paths.extend((root/'src/linn'/folder).glob('*.py'))
    source_paths.append(root/'src/linn/experiments/company2_case.py')
    raw_paths = [p for p in (root/config['data_dir']).iterdir() if p.is_file()]
    original_hashes = {str(p.relative_to(root)): sha256(p) for p in source_paths + raw_paths}
    signature = {'settings': FROZEN, 'source_and_raw_hashes': original_hashes}
    signature_path = out/'protocol.json'
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError('Frozen protocol/code/input changed; refuse mixing cached results')
    else:
        save_json(signature_path, signature)
    save_json(out/'environment.json', {'python': platform.python_version(), 'platform': platform.platform(),
              'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__,
              'torch_num_threads': torch.get_num_threads()})
    raw, _, _ = load_company2(root/config['data_dir'])
    frame, _ = prepare_casting_case(raw, config)
    original, _, _ = select_case(frame, FROZEN['selection'])
    original_ids = set(original.loc[original.split.eq('test'), 'record_id'])
    fold_audits, counts_rows, observed_rows = [], [], []
    tolerance = FROZEN['rounding_tolerance_hours']
    for month in FROZEN['test_months']:
        fold = month[:7]
        split, boundaries = monthly_split(frame, month, config['period_start'])
        audit = {'fold': fold, **boundaries}
        try:
            selected, machines, counts = select_case(split, FROZEN['selection'])
        except ValueError as error:
            audit.update(status='skipped', reason=str(error))
            fold_audits.append(audit)
            print(f'{fold}: skipped, {error}', flush=True)
            continue
        counts.reset_index().assign(fold=fold).to_csv(out/'private'/f'{fold}_machine_selection.csv', index=False)
        for split_name in ['train', 'validation', 'test']:
            group = selected[selected.split.eq(split_name)]
            counts_rows.append({'fold': fold, 'split': split_name, 'n_operations': len(group),
                                'n_orders': group.order_id.nunique(), 'n_machines': group.machine.nunique()})
        audit.update(selected_machines=machines,
                     n_train=int(selected.split.eq('train').sum()),
                     n_validation=int(selected.split.eq('validation').sum()),
                     n_test=int(selected.split.eq('test').sum()),
                     n_test_orders=int(selected.loc[selected.split.eq('test'), 'order_id'].nunique()),
                     n_purged_historical=int((selected.split.eq('purged') & selected.global_order_first.lt(boundaries['test_end_exclusive'])).sum()))
        if not audit['n_test']:
            audit.update(status='skipped', reason='No eligible complete test operations after past-only selection')
            fold_audits.append(audit)
            continue
        if min(int(audit['n_train']*.1), int(audit['n_validation']*.1)) < 1:
            audit.update(status='skipped', reason='10% labels leave an empty training/validation target channel')
            fold_audits.append(audit)
            continue
        audit['status'] = 'completed'
        fold_audits.append(audit)
        selected.to_csv(out/'private'/f'{fold}_prepared_operations.csv', index=False)
        test_records = selected[selected.split.eq('test')]
        for level in FROZEN['completeness']:
            for seed in FROZEN['evaluation_seeds']:
                prefix = f'{fold}_{int(100*level):03d}_{seed}'
                result_path = out/'private'/f'{prefix}.json'
                if result_path.exists():
                    observed_rows.extend(json.loads(result_path.read_text())['observed_label_counts'])
                    continue
                ds, encoder, spec, masks = prepare_development(selected, level, seed, tolerance=tolerance)
                masks.to_csv(out/'private'/f'{prefix}_masks.csv', index=False)
                save_json(out/'private'/f'{prefix}_preprocessing.json', {'encoder': encoder.metadata(), 'context': spec.context})
                ds['test'] = WorkstationDataset(test_records, encoder)
                label_counts = [{'fold': fold, 'completeness': level, 'seed': seed, 'split': split_name,
                                 'target': target, 'n_observed': int(ds[split_name].mask[:,j].sum())}
                                for split_name in ['train', 'validation'] for j, target in enumerate(TARGETS)]
                observed_rows.extend(label_counts)
                ridge = baseline_predictions(ds['train'], ds['test'], 'quantity_linear', FROZEN['ridge_alpha'])
                predictions = {'ridge': ridge, 'ridge_projection': project_durations(ridge, tolerance)}
                infos = {}
                for method in ['data_driven', 'domain_informed']:
                    weight = FROZEN['constraint_weight'] if method == 'domain_informed' else 0.
                    trainer, history, info = train_candidate(ds['train'], ds['validation'], spec,
                                                            FROZEN['candidate'], weight, seed, FROZEN)
                    predictions[method] = predict(trainer, ds['test'])
                    pd.DataFrame(history).to_csv(out/'private'/f'{prefix}_{method}_history.csv', index=False)
                    infos[method] = info
                predictions['data_driven_projection'] = project_durations(predictions['data_driven'], tolerance)
                frames = []
                for method, p in predictions.items():
                    if method.endswith('_projection'):
                        assert workstation_violations(p, tolerance)['any_violation_fraction'] == 0.
                    frames.append(pd.DataFrame({'fold': fold, 'record_id': test_records.record_id.to_numpy(),
                        'order_id': test_records.order_id.to_numpy(), 'machine': test_records.machine.to_numpy(),
                        'completeness': level, 'seed': seed, 'method': method,
                        'elapsed_observed': ds['test'].y[:,0].numpy(), 'booked_observed': ds['test'].y[:,1].numpy(),
                        'elapsed_predicted': p[:,0], 'booked_predicted': p[:,1]}))
                pd.concat(frames, ignore_index=True).to_csv(out/'private'/f'{prefix}_predictions.csv', index=False)
                save_json(result_path, {'observed_label_counts': label_counts, 'neural_fits': infos})
                print(f'Completed fold {fold}, labels {level:.0%}, seed {seed}, test n={len(test_records)}', flush=True)
    save_json(out/'fold_audit.json', fold_audits)
    pd.DataFrame(counts_rows).to_csv(out/'split_counts.csv', index=False)
    pd.DataFrame(observed_rows).to_csv(out/'observed_label_counts.csv', index=False)
    assert all(sha256(root/p) == digest for p, digest in original_hashes.items())
    save_json(out/'validation.json', {'raw_and_source_files_unchanged': True, 'test_tuning': False,
              'configuration_has_later_period_calibration': True,
              'independent_industrial_confirmation': False,
              'completed_neural_fits': len(list((out/'private').glob('*_history.csv')))})
    return summarize(out, fold_audits, original_ids, tolerance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[3]))
    run(parser.parse_args().root)
