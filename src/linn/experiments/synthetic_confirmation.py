"""Frozen paired confirmation with sparse training AND validation observations."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch

from linn.data.loaders import sha256
from linn.data.simulation import simulate_workstations
from linn.data.workstation_dataset import FeatureEncoder, WorkstationDataset, TARGETS
from linn.evaluation.degradation import observation_mask
from linn.evaluation.metrics import regression_metrics, workstation_violations
from linn.experiments.workstation_ms1 import save_json
from linn.processes.workstation import workstation_spec
from linn.training.tuning import train_candidate, predict


def prepare_sparse(frame, completeness, seed):
    keep = ['record_id', 'order_id', 'machine', 'product', 'operation', 'start',
            'planned_quantity', 'split', *TARGETS]
    parts = {s: frame.loc[frame.split.eq(s), keep].copy()
             for s in ['train', 'validation', 'test']}
    audit, masks = [], {}
    for i, (split, part) in enumerate(parts.items()):
        mask = (np.ones((len(part), 2), dtype=bool) if split == 'test' else
                observation_mask(part.record_id, completeness, seed + i*100003, 2))
        if not mask.any(axis=0).all():
            raise ValueError('Each development target needs at least one observed label')
        masks[split] = mask
        for j, target in enumerate(TARGETS):
            part.loc[~mask[:, j], target] = np.nan
        part['quantity_missing'] = 0.
        audit.extend({'record_id': rid, 'split': split, 'elapsed_observed': bool(m[0]),
                      'booked_observed': bool(m[1])} for rid, m in zip(part.record_id, mask))
    encoder = FeatureEncoder(['planned_quantity', 'quantity_missing']).fit(parts['train'])
    spec = workstation_spec(parts['train'], 0.)
    datasets = {s: WorkstationDataset(p, encoder, masks[s]) for s, p in parts.items()}
    return datasets, spec, pd.DataFrame(audit), {'encoder': encoder.metadata(), 'context': spec.context}


def run(config):
    path = Path(config).resolve(); root = path.parent.parent
    settings = json.loads(path.read_text())
    base = json.loads((root/settings['simulation_config']).read_text())
    choices = json.loads((root/settings['frozen_choices']).read_text())
    out = (root/settings['output_dir']).resolve()
    if out != (root/'results/synthetic_confirmation').resolve():
        raise ValueError('Use dedicated confirmation output directory')
    out.mkdir(parents=True, exist_ok=True)
    core = [p for folder in ['data', 'models', 'constraints', 'processes', 'training']
            for p in (root/'src/linn'/folder).glob('*.py')]
    core += [Path(__file__), root/'src/linn/evaluation/degradation.py',
             root/'src/linn/evaluation/metrics.py', root/'docs/synthetic_confirmation_protocol.md']
    signature = {'settings': settings, 'generator': base, 'choices': choices,
                 'hashes': {str(p.relative_to(root)): sha256(p) for p in core}}
    protocol = out/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != signature:
        raise ValueError('Frozen protocol/code changed; cached fits cannot be mixed')
    if not protocol.exists():
        save_json(protocol, signature)
    save_json(out/'environment.json', {'python': platform.python_version(), 'platform': platform.platform(),
              'packages': {n: importlib.import_module(n).__version__
                           for n in ['numpy', 'pandas', 'torch', 'sklearn', 'matplotlib']}})
    protected = [p for folder in ['daten', 'results/ms2', 'results/synthetic', 'results/synthetic_degradation']
                 for p in (root/folder).rglob('*') if p.is_file()]
    before = {str(p.relative_to(root)): sha256(p) for p in protected}
    if (out/'protected_hashes.json').exists():
        assert json.loads((out/'protected_hashes.json').read_text()) == before
    else:
        save_json(out/'protected_hashes.json', before)
    for folder in ['references', 'conditions', 'runs']:
        (out/folder).mkdir(exist_ok=True)
    total = len(settings['simulation_seeds'])*len(base['simulation']['machines'])*len(settings['label_completeness'])*2
    count = 0
    for world in settings['simulation_seeds']:
        frame = simulate_workstations(dict(base['simulation'], seed=world))
        frame.to_csv(out/'references'/f'world_{world}.csv', index=False)
        seed = 900000 + world
        for mi, (machine, group) in enumerate(frame.groupby('machine')):
            choice = choices[machine+':full']
            for level in settings['label_completeness']:
                key = f'{world}_{machine}_{int(level*100):03d}'
                ds, spec, masks, meta = prepare_sparse(group, level, seed+mi*100003)
                masks.to_csv(out/'conditions'/f'{key}_mask.csv', index=False)
                save_json(out/'conditions'/f'{key}_preprocessing.json', meta)
                for method in ['data_driven', 'domain_informed']:
                    name = key+'_'+method
                    target = out/'runs'/f'{name}.json'
                    if target.exists():
                        count += 1
                        continue
                    weight = choice['constraint_weight'] if method == 'domain_informed' else 0.
                    trainer, history, info = train_candidate(ds['train'], ds['validation'], spec,
                                                            choice['candidate'], weight, seed, settings)
                    repeat = world == settings['simulation_seeds'][0] and mi == 0 and level == .05
                    if repeat:
                        again, h2, i2 = train_candidate(ds['train'], ds['validation'], spec,
                                                       choice['candidate'], weight, seed, settings)
                        assert history == h2 and info == i2
                        assert all(torch.equal(v, again.model.state_dict()[k])
                                   for k, v in trainer.model.state_dict().items())
                    p = predict(trainer, ds['test']); y = ds['test'].y.numpy()
                    violation = workstation_violations(p, 0.)
                    common = {'world': world, 'machine': machine, 'seed': seed,
                              'completeness': level, 'method': method, 'n_train': len(ds['train']),
                              'n_test': len(ds['test']), 'constraint_violation': violation['any_violation_fraction'],
                              'epoch_budget_reached': len(history) == settings['max_epochs'], **info}
                    for split in ['train', 'validation']:
                        for j, t in enumerate(TARGETS):
                            common[f'n_{split}_{t}'] = int(ds[split].mask[:, j].sum())
                    rows = [{**common, 'target': t, **regression_metrics(y[:, j], p[:, j])}
                            for j, t in enumerate(TARGETS)]
                    pd.DataFrame({'record_id': group[group.split.eq('test')].record_id.to_numpy(),
                                  'elapsed_observed': y[:, 0], 'booked_observed': y[:, 1],
                                  'elapsed_predicted': p[:, 0], 'booked_predicted': p[:, 1]}).to_csv(
                                      out/'runs'/f'{name}_predictions.csv', index=False)
                    pd.DataFrame(history).to_csv(out/'runs'/f'{name}_history.csv', index=False)
                    digest = hashlib.sha256(b''.join(v.detach().numpy().tobytes()
                                                    for v in trainer.model.state_dict().values())).hexdigest()
                    save_json(target, {'metrics': rows, 'exact_repeat_checked': repeat, 'parameter_sha256': digest})
                    count += 1
                print(f'{count}/{total}: world {world}, {machine}, {level:.0%}', flush=True)
    assert all(sha256(root/p) == h for p, h in before.items())
    save_json(out/'validation.json', {'fits': count, 'protected_unchanged': True,
                                    'sparse_validation': True, 'test_tuning': False, 'industrial_ms3': False})
    from linn.evaluation.confirmation_report import summarize
    summarize(out, settings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/synthetic_confirmation.json')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        from linn.evaluation.confirmation_report import summarize
        config = Path(args.config).resolve(); settings = json.loads(config.read_text())
        summarize(config.parent.parent/settings['output_dir'], settings)
    else:
        run(args.config)
