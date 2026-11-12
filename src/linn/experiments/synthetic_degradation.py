"""Paired synthetic label/input degradation with fresh simulated worlds and frozen choices."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import torch
from linn.data.simulation import simulate_workstations
from linn.data.loaders import sha256
from linn.data.workstation_dataset import FeatureEncoder, WorkstationDataset, TARGETS
from linn.processes.workstation import workstation_spec
from linn.training.tuning import train_candidate, predict
from linn.evaluation.degradation import observation_mask, ObservedMedianImputer
from linn.evaluation.metrics import regression_metrics, workstation_violations
from linn.experiments.workstation_ms1 import save_json


def prepare_condition(frame, mechanism, completeness, mask_seed):
    """No truth/oracle/end columns survive into the training frame."""
    keep=['record_id','order_id','machine','product','operation','start','planned_quantity','split']+list(TARGETS)
    parts={s:frame.loc[frame['split'].eq(s),keep].copy() for s in ['train','validation','test']}
    masks={s:np.ones((len(f),2),dtype=bool) for s,f in parts.items()}
    quantity_masks={s:np.ones(len(f),dtype=bool) for s,f in parts.items()}
    if mechanism=='labels':
        masks['train']=observation_mask(parts['train'].record_id,completeness,mask_seed,2)
    elif mechanism=='quantity':
        for i,(s,f) in enumerate(parts.items()):
            quantity_masks[s]=observation_mask(f.record_id,completeness,mask_seed+10000*(i+1))[:,0]
    else:
        raise ValueError('Unknown degradation mechanism')
    for s,f in parts.items():
        for j,col in enumerate(TARGETS):
            f.loc[~masks[s][:,j],col]=np.nan
    imputer=ObservedMedianImputer().fit(parts['train'].planned_quantity,quantity_masks['train'])
    audit=[]
    for s,f in parts.items():
        f['planned_quantity'],f['quantity_missing']=imputer.transform(f.planned_quantity,quantity_masks[s])
        for record,lmask,qmask in zip(f.record_id,masks[s],quantity_masks[s]):
            audit.append({'record_id':record,'split':s,'elapsed_observed':bool(lmask[0]),
                          'booked_observed':bool(lmask[1]),'quantity_observed':bool(qmask)})
    encoder=FeatureEncoder(['planned_quantity','quantity_missing']).fit(parts['train'])
    spec=workstation_spec(parts['train'],0.)
    datasets={s:WorkstationDataset(f,encoder,target_mask=masks[s]) for s,f in parts.items()}
    return datasets,spec,pd.DataFrame(audit),{'encoder':encoder.metadata(),'context':spec.context,'quantity_imputation_median':imputer.median}


def run(config_path):
    config_path=Path(config_path).resolve();root=config_path.parent.parent
    settings=json.loads(config_path.read_text());base=json.loads((root/settings['simulation_config']).read_text())
    choices=json.loads((root/settings['frozen_choices']).read_text())
    out=(root/settings['output_dir']).resolve()
    if out != (root/'results/synthetic_degradation').resolve():
        raise ValueError('Use the dedicated synthetic_degradation output directory')
    out.mkdir(parents=True,exist_ok=True)
    core=[root/'src/linn/training/tuning.py',root/'src/linn/training/trainer.py',root/'src/linn/training/losses.py',
          root/'src/linn/data/workstation_dataset.py',root/'src/linn/data/simulation.py',root/'src/linn/evaluation/degradation.py',
          root/'src/linn/models/neural_model.py',root/'src/linn/constraints/workstation_constraints.py',
          root/'src/linn/processes/workstation.py',Path(__file__)]
    signature={'settings':settings,'simulation_config':base,'frozen_choices':choices,
               'core_hashes':{str(p.relative_to(root)):sha256(p) for p in core}}
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=signature:
            raise ValueError('Protocol/code changed; do not mix cached runs. Use the original protocol or archive the run first.')
    else:save_json(out/'protocol.json',signature)
    save_json(out/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
              'packages':{name:importlib.import_module(name).__version__ for name in ['numpy','pandas','torch','sklearn','matplotlib']}})
    protected=list((root/'daten').glob('*'))+list((root/'results/ms2').rglob('*'))+list((root/'results/synthetic').rglob('*'))
    raw_hashes={str(p):sha256(p) for p in protected if p.is_file()}
    for directory in ['runs','conditions','references']:(out/directory).mkdir(exist_ok=True)
    completed=0
    for world in settings['simulation_seeds']:
        simulation=dict(base['simulation'],seed=world)
        frame=simulate_workstations(simulation)
        frame.to_csv(out/'references'/f'world_{world}.csv',index=False)
        for machine_i,(machine,g) in enumerate(frame.groupby('machine')):
            choice=choices[machine+':full']
            for seed in settings['paired_seeds']:
                mask_seed=world*1009+seed*17+machine_i*100003
                conditions=[('labels',c) for c in settings['label_completeness']]
                conditions += [('quantity',c) for c in settings['quantity_completeness'] if c!=1.]
                for mechanism,level in conditions:
                    condition=f'{world}_{machine}_{seed}_{mechanism}_{int(level*100):03d}'
                    datasets,spec,masks,metadata=prepare_condition(g,mechanism,level,mask_seed)
                    masks.to_csv(out/'conditions'/f'{condition}_mask.csv',index=False)
                    save_json(out/'conditions'/f'{condition}_preprocessing.json',metadata)
                    for method in ['data_driven','domain_informed']:
                        key=condition+'_'+method;result_path=out/'runs'/f'{key}.json'
                        if result_path.exists():
                            completed+=1
                            continue
                        weight=choice['constraint_weight'] if method=='domain_informed' else 0.
                        trainer,history,info=train_candidate(datasets['train'],datasets['validation'],spec,choice['candidate'],weight,seed,settings)
                        repeat_checked=False
                        if world==settings['simulation_seeds'][0] and seed==settings['paired_seeds'][0] and mechanism=='labels' and level==.05:
                            again,h2,i2=train_candidate(datasets['train'],datasets['validation'],spec,choice['candidate'],weight,seed,settings)
                            assert history==h2 and info==i2
                            assert all(torch.equal(v,again.model.state_dict()[k]) for k,v in trainer.model.state_dict().items())
                            repeat_checked=True
                        p=predict(trainer,datasets['test']);v=workstation_violations(p,0.)
                        counts=datasets['train'].mask.sum(dim=0).tolist()
                        common={'simulation_seed':world,'machine':machine,'seed':seed,'mask_seed':mask_seed,'mechanism':mechanism,
                                'completeness':level,'method':method,'n_train':len(datasets['train']),
                                'n_observed_elapsed':counts[0],'n_observed_booked':counts[1],
                                'n_validation':len(datasets['validation']),'n_test':len(datasets['test']),
                                'constraint_violation':v['any_violation_fraction'],
                                'mean_accounting_excess_hours':v['mean_violation_hours'][2],**info}
                        metrics=[{**common,'target':target,**regression_metrics(datasets['test'].y.numpy()[:,j],p[:,j])} for j,target in enumerate(TARGETS)]
                        pred=pd.DataFrame({'record_id':g[g['split']=='test'].record_id.to_numpy(),
                                           'elapsed_observed':datasets['test'].y[:,0].numpy(),'booked_observed':datasets['test'].y[:,1].numpy(),
                                           'elapsed_predicted':p[:,0],'booked_predicted':p[:,1]})
                        pred.to_csv(out/'runs'/f'{key}_predictions.csv',index=False)
                        pd.DataFrame(history).to_csv(out/'runs'/f'{key}_history.csv',index=False)
                        digest=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in trainer.model.state_dict().values())).hexdigest()
                        save_json(result_path,{'metrics':metrics,'exact_repeat_checked':repeat_checked,'parameter_sha256':digest})
                        completed+=1
                    print(f'{completed}/288 fits: world {world}, {machine}, seed {seed}, {mechanism}, {level:.0%}',flush=True)
    assert all(sha256(p)==h for p,h in raw_hashes.items())
    save_json(out/'validation.json',{'protected_raw_and_prior_results_unchanged':True,'fit_count':completed,
              'test_used_for_tuning':False,'clean_validation_anchor':True,'industrial_ms3_run':False})
    summarize(out,settings)


def summarize(out,settings):
    from linn.evaluation.degradation_report import summarize_results
    summarize_results(out,settings)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/synthetic_degradation.json')
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.summarize_only:
        cfg=Path(args.config).resolve();settings=json.loads(cfg.read_text());summarize(cfg.parent.parent/settings['output_dir'],settings)
    else:run(args.config)
