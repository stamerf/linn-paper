"""Exploratory industrial reconstruction with independently retained confirmed quantity."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from linn.data.company2_case import prepare_casting_case, Company2CaseEncoder
from linn.data.loaders import load_company2, sha256
from linn.data.workstation_dataset import WorkstationDataset, TARGETS
from linn.evaluation.degradation import observation_mask
from linn.evaluation.metrics import regression_metrics, workstation_violations
from linn.experiments.workstation_ms1 import save_json
from linn.processes.workstation import workstation_spec
from linn.training.tuning import tune, train_candidate, predict


def select_case(frame, settings):
    counts = frame.groupby(['machine', 'split']).size().unstack(fill_value=0)
    eligible = ((counts.get('train', 0) >= settings['min_train']) &
                (counts.get('validation', 0) >= settings['min_validation']))
    selected = sorted(counts.index[eligible].tolist())
    if not selected:
        raise ValueError('No machines satisfy the fixed development-volume rule')
    counts['selected'] = eligible
    return frame[frame.machine.isin(selected)].copy(), selected, counts


def prepare_development(frame, completeness, seed, include_quantity=True, tolerance=.02):
    """Strip hidden duration labels BEFORE scales and tensors; no test dataset here."""
    parts = {s: frame[frame.split.eq(s)].copy() for s in ['train', 'validation']}
    masks, audit = {}, []
    for i, (split, part) in enumerate(parts.items()):
        masks[split] = observation_mask(part.record_id, completeness, seed+i*100003, 2)
        if not masks[split].any(axis=0).all():
            raise ValueError('At least one observed training and validation value per target required')
        for j, name in enumerate(TARGETS):
            part.loc[~masks[split][:, j], name] = np.nan
        audit.extend({'record_id': rid, 'split': split, 'elapsed_observed': bool(m[0]),
                      'booked_observed': bool(m[1])} for rid, m in zip(part.record_id, masks[split]))
    encoder = Company2CaseEncoder(include_quantity=include_quantity).fit(parts['train'])
    spec = workstation_spec(parts['train'], tolerance)
    datasets = {s: WorkstationDataset(p, encoder, masks[s]) for s, p in parts.items()}
    return datasets, encoder, spec, pd.DataFrame(audit)


def baseline_predictions(train, evaluation, method, alpha=1.):
    result = []
    for j in range(train.y.shape[1]):
        mask = train.mask[:, j].numpy()
        y = train.y[:, j].numpy()[mask]
        if not len(y):
            raise ValueError('Baseline requires observed labels')
        if method == 'median':
            p = np.full(len(evaluation), np.median(y))
        elif method == 'quantity_linear':
            model = Ridge(alpha=alpha, solver='svd').fit(train.x.numpy()[mask], y)
            p = model.predict(evaluation.x.numpy())
        else:
            raise ValueError('Unknown baseline')
        result.append(p)
    return np.column_stack(result)


def run(config_path):
    path = Path(config_path).resolve(); root = path.parent.parent
    settings = json.loads(path.read_text())
    config = json.loads((root/settings['preprocessing_config']).read_text())
    out = (root/settings['output_dir']).resolve()
    if out != (root/'results/company2_case').resolve():
        raise ValueError('Use the dedicated Company 2 case directory')
    out.mkdir(parents=True, exist_ok=True)
    protected = [p for folder in ['daten', 'results/ms1', 'results/ms2', 'results/synthetic',
                                  'results/synthetic_degradation', 'results/synthetic_confirmation']
                 for p in (root/folder).rglob('*') if p.is_file()]
    original = {str(p.relative_to(root)): sha256(p) for p in protected}
    if (out/'protected_hashes.json').exists():
        assert json.loads((out/'protected_hashes.json').read_text()) == original
    else:
        save_json(out/'protected_hashes.json', original)
    tables, _, inventory = load_company2(root/config['data_dir'])
    frame, audit = prepare_casting_case(tables, config)
    frame, selected, selection = select_case(frame, settings['selection'])
    assert frame.groupby('order_id').split.nunique().max() == 1
    selection.to_csv(out/'machine_selection.csv')
    audit.to_csv(out/'cohort_audit.csv', index=False)
    frame.to_csv(out/'prepared_operations.csv', index=False)
    summary = {'selected_machines': selected, 'selection': settings['selection'],
               'splits': {s: {'n_rows': len(g), 'n_orders': int(g.order_id.nunique()),
                              'machines': g.groupby('machine').size().to_dict()}
                          for s, g in frame.groupby('split')},
               'exclusions': audit.attrs['exclusion_counts'],
               'prior_test_machines': settings['prior_test_machines'],
               'test_previously_used_in_part': True,
               'quantity_role': 'Retrospective confirmed good plus scrap, independently retained',
               'target_role': 'Cumulative elapsed booking-window hours and booked processing hours',
               'coupled_order_identity': 'Unavailable: explicit parent-child columns all empty',
               'sparse_configuration_calibration': 'Complete training and validation labels'}
    save_json(out/'cohort_summary.json', summary)
    save_json(out/'raw_inventory.json', inventory)
    core = [p for folder in ['models', 'constraints', 'processes', 'training']
            for p in (root/'src/linn'/folder).glob('*.py')]
    core += [root/'src/linn/data'/n for n in ['company2_case.py', 'loaders.py', 'preprocessing.py', 'workstation_dataset.py']]
    core += [Path(__file__), root/'src/linn/evaluation/degradation.py', root/'src/linn/evaluation/metrics.py']
    signature = {'settings': settings, 'preprocessing': config, 'machines': selected,
                 'hashes': {str(p.relative_to(root)): sha256(p) for p in core},
                 'raw_hashes': {k: v['sha256'] for k, v in inventory.items() if isinstance(v, dict)}}
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != signature:
            raise ValueError('Protocol/code changed: cannot mix cached experiments')
    else:
        save_json(out/'protocol.json', signature)
    save_json(out/'environment.json', {'python': platform.python_version(), 'platform': platform.platform(),
              'packages': {n: importlib.import_module(n).__version__ for n in ['numpy', 'pandas', 'torch', 'sklearn', 'matplotlib']}})
    datasets, encoder, spec, _ = prepare_development(frame, 1., 17, tolerance=settings['rounding_tolerance_hours'])
    if (out/'selected_configs.json').exists():
        choice = json.loads((out/'selected_configs.json').read_text())
    else:
        print('Select configuration using complete training/validation labels only', flush=True)
        choice, trials = tune(datasets['train'], datasets['validation'], spec, settings)
        pd.DataFrame(trials).to_csv(out/'tuning_trials.csv', index=False)
        save_json(out/'selected_configs.json', choice)
    selection_hash = sha256(out/'selected_configs.json')
    for folder in ['runs', 'conditions', 'models']:
        (out/folder).mkdir(exist_ok=True)
    # Test tensors/predictions are first constructed below, after all choices freeze.
    for level in settings['completeness']:
        for seed in settings['evaluation_seeds']:
            for include_quantity in ([True, False] if level == 1. else [True]):
                ds, encoder, spec, masks = prepare_development(frame, level, seed, include_quantity,
                                                               settings['rounding_tolerance_hours'])
                prefix = f'{int(100*level):03d}_{seed}_q{int(include_quantity)}'
                masks.to_csv(out/'conditions'/f'{prefix}_mask.csv', index=False)
                save_json(out/'conditions'/f'{prefix}_preprocessing.json', {'encoder': encoder.metadata(), 'context': spec.context})
                ds['test'] = WorkstationDataset(frame[frame.split.eq('test')], encoder)
                methods = ['median', 'quantity_linear', 'data_driven', 'domain_informed'] if include_quantity else ['without_quantity']
                for method in methods:
                    name = f'{prefix}_{method}'; result_path = out/'runs'/f'{name}.json'
                    if result_path.exists():
                        continue
                    history, info, repeat = [], {}, False
                    trainer = None
                    if method not in ['median', 'quantity_linear']:
                        weight = choice['constraint_weight'] if method == 'domain_informed' else 0.
                        trainer, history, info = train_candidate(ds['train'], ds['validation'], spec,
                                                                choice['candidate'], weight, seed, settings)
                        if level == .1 and seed == settings['evaluation_seeds'][0]:
                            again, h2, i2 = train_candidate(ds['train'], ds['validation'], spec,
                                                           choice['candidate'], weight, seed, settings)
                            assert history == h2 and info == i2
                            assert all(torch.equal(v, again.model.state_dict()[k]) for k, v in trainer.model.state_dict().items())
                            repeat = True
                        torch.save({'state_dict': trainer.model.state_dict(), 'input_dim': ds['train'].x.shape[1],
                                    'output_dim': 2, 'candidate': choice['candidate'], 'context': spec.context},
                                   out/'models'/f'{name}.pt')
                    rows, predictions = [], []
                    for split in ['validation', 'test']:
                        p = (predict(trainer, ds[split]) if trainer else
                             baseline_predictions(ds['train'], ds[split], method, settings['ridge_alpha']))
                        v = workstation_violations(p, settings['rounding_tolerance_hours'])
                        common = {'machine': 'pooled', 'method': method, 'completeness': level, 'seed': seed,
                                  'split': split, 'n_train': len(ds['train']), 'n_test': len(ds['test']),
                                  'n_observed_train_per_target': int(ds['train'].mask[:, 0].sum()),
                                  'n_observed_validation_per_target': int(ds['validation'].mask[:, 0].sum()),
                                  'constraint_violation': v['any_violation_fraction'], **info}
                        for j, target in enumerate(TARGETS):
                            mask = ds[split].mask[:, j].numpy()
                            rows.append({**common, 'target': target,
                                         **regression_metrics(ds[split].y[:, j].numpy()[mask], p[mask, j])})
                        records = frame[frame.split.eq(split)]
                        predictions.append(pd.DataFrame({'record_id': records.record_id.to_numpy(), 'order_id': records.order_id.to_numpy(),
                            'machine': records.machine.to_numpy(), 'split': split, 'method': method, 'completeness': level, 'seed': seed,
                            'elapsed_observed': np.where(ds[split].mask[:, 0].numpy(), ds[split].y[:, 0].numpy(), np.nan),
                            'booked_observed': np.where(ds[split].mask[:, 1].numpy(), ds[split].y[:, 1].numpy(), np.nan),
                            'elapsed_predicted': p[:, 0], 'booked_predicted': p[:, 1]}))
                    pd.concat(predictions).to_csv(out/'runs'/f'{name}_predictions.csv', index=False)
                    if history:
                        pd.DataFrame(history).assign(completeness=level, seed=seed, method=method).to_csv(out/'runs'/f'{name}_history.csv', index=False)
                    save_json(result_path, {'metrics': rows, 'exact_repeat_checked': repeat})
                print(f'Completed {level:.0%}, seed {seed}, quantity={include_quantity}', flush=True)
    assert sha256(out/'selected_configs.json') == selection_hash
    assert all(sha256(root/p) == h for p, h in original.items())
    results = [json.loads(p.read_text()) for p in sorted((out/'runs').glob('*.json'))]
    pd.DataFrame([row for r in results for row in r['metrics']]).to_csv(out/'metrics.csv', index=False)
    pd.concat([pd.read_csv(p) for p in sorted((out/'runs').glob('*_predictions.csv'))], ignore_index=True).to_csv(out/'predictions.csv', index=False)
    pd.concat([pd.read_csv(p) for p in sorted((out/'runs').glob('*_history.csv'))], ignore_index=True).to_csv(out/'history.csv', index=False)
    save_json(out/'validation.json', {'protected_unchanged': True, 'test_tuning': False,
              'selected_config_sha256': selection_hash, 'completed_method_runs': len(results),
              'exact_repeat_checks': sum(r['exact_repeat_checked'] for r in results),
              'primary_case_exploratory': True, 'full_label_configuration_calibration': True})
    from linn.evaluation.company2_case_report import summarize
    summarize(out, settings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/company2_case.json')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        from linn.evaluation.company2_case_report import summarize
        p = Path(args.config).resolve(); settings = json.loads(p.read_text())
        summarize(p.parent.parent/settings['output_dir'], settings)
    else:
        run(args.config)
