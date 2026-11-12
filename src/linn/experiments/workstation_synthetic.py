"""Controlled synthetic diagnostic; separate from industrial MS2 and missing-data MS3."""
import argparse
import importlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from linn.data.simulation import simulate_workstations
from linn.data.workstation_dataset import FeatureEncoder,WorkstationDataset,TARGETS
from linn.data.loaders import sha256
from linn.processes.workstation import workstation_spec
from linn.training.tuning import tune,train_candidate,predict
from linn.evaluation.metrics import regression_metrics,workstation_violations
from linn.evaluation.synthetic_plots import plot_synthetic
from linn.experiments.workstation_ms1 import save_json


def tune_limited(train,validation,spec,settings):
    trials=[]
    for i,candidate in enumerate(settings["candidates"]):
        for seed in settings["tuning_seeds"]:
            _,_,info=train_candidate(train,validation,spec,candidate,0.,seed,settings)
            trials.append({"stage":"architecture","candidate":i,"weight":0.,"seed":seed,**info})
    scores={i:np.mean([r["validation_score"] for r in trials if r["candidate"]==i]) for i in range(len(settings["candidates"]))}
    best=min(scores,key=lambda i:(scores[i],i))
    return {"candidate":settings["candidates"][best],"candidate_index":best},trials


def run(config_path):
    config_path=Path(config_path).resolve();root=config_path.parent.parent
    settings=json.loads(config_path.read_text())
    out=(root/settings["output_dir"]).resolve()
    if out.parent != (root/"results").resolve() or out.name in ["ms1","ms2","ms3"]:
        raise ValueError("Synthetic outputs must use a separate immediate results subdirectory")
    out.mkdir(parents=True,exist_ok=True)
    protected=list((root/"daten").glob("*"))+list((root/"results/ms2").rglob("*"))
    before={str(p):sha256(p) for p in protected if p.is_file()}
    frame=simulate_workstations(settings["simulation"])
    pd.testing.assert_frame_equal(frame,simulate_workstations(settings["simulation"]))
    frame.to_csv(out/"simulated_bookings.csv",index=False)
    save_json(out/"config.json",settings)
    save_json(out/"environment.json",{"python":platform.python_version(),"platform":platform.platform(),
              "packages":{name:importlib.import_module(name).__version__ for name in ["numpy","pandas","torch","sklearn","matplotlib"]}})
    frame.groupby(["machine","split"]).agg(n_jobs=("record_id","size"),mean_elapsed=("elapsed_hours","mean"),
                      std_elapsed=("elapsed_hours","std"),mean_queue_wait=("truth_queue_wait_hours","mean")).to_csv(out/"simulation_summary.csv")
    selected,prepared,trials={},{},[]
    for machine,g in frame.groupby("machine"):
        train=g[g["split"]=="train"];val=g[g["split"]=="validation"]
        assert train.end.max()<=val.start.min()
        spec=workstation_spec(train,settings["rounding_tolerance_hours"])
        entries={}
        for view,columns in [("full",settings["numeric_features"]),("limited",[])]:
            encoder=FeatureEncoder(columns).fit(train)
            ds_train=WorkstationDataset(train,encoder);ds_val=WorkstationDataset(val,encoder)
            print(f"Tuning {machine}, {view} features (validation only)",flush=True)
            choice,rows=(tune if view=="full" else tune_limited)(ds_train,ds_val,spec,settings)
            selected[machine+":"+view]=choice
            trials.extend({"machine":machine,"feature_view":view,**r} for r in rows)
            entries[view]=(encoder,ds_train,ds_val)
            save_json(out/f"{machine}_{view}_encoder.json",encoder.metadata())
        prepared[machine]=(g,spec,entries)
    save_json(out/"selected_configs.json",selected)
    selection_hash=sha256(out/"selected_configs.json")
    pd.DataFrame(trials).to_csv(out/"tuning_trials.csv",index=False)
    metrics,predictions,histories,diagnostics=[],[],[],[]
    for machine,(g,spec,entries) in prepared.items():
        parts={phase:g[g["split"]==phase] for phase in ["train","validation","test"]}
        assert parts["validation"].end.max()<=parts["test"].start.min()
        median=parts["train"][list(TARGETS)].median().to_numpy()
        for seed in settings["evaluation_seeds"]:
            for method in ["median","limited_features","data_driven","domain_informed","oracle_mean"]:
                trainer=None
                view="limited" if method=="limited_features" else "full"
                encoder,ds_train,ds_val=entries[view]
                if method not in ["median","oracle_mean"]:
                    choice=selected[machine+":"+view]
                    weight=choice["constraint_weight"] if method=="domain_informed" else 0.
                    trainer,history,info=train_candidate(ds_train,ds_val,spec,choice["candidate"],weight,seed,settings)
                    histories.extend({"machine":machine,"method":method,"seed":seed,**row} for row in history)
                    diagnostics.append({"machine":machine,"method":method,"seed":seed,**info})
                    if seed==settings["evaluation_seeds"][0] and method!="limited_features":
                        other,history2,info2=train_candidate(ds_train,ds_val,spec,choice["candidate"],weight,seed,settings)
                        assert history==history2 and info==info2
                        assert all(torch.equal(v,other.model.state_dict()[k]) for k,v in trainer.model.state_dict().items())
                    models=out/"models";models.mkdir(exist_ok=True)
                    torch.save({"state_dict":trainer.model.state_dict(),"context":spec.context,"candidate":choice["candidate"],
                                "input_dim":ds_train.x.shape[1],"output_dim":2,"numeric_features":list(encoder.numeric_columns)},
                               models/f"{machine}_{method}_{seed}.pt")
                for phase,f in parts.items():
                    ds=WorkstationDataset(f,encoder)
                    if method=="median":p=np.tile(median,(len(f),1))
                    elif method=="oracle_mean":p=f[["oracle_elapsed_hours","oracle_booked_hours"]].to_numpy()
                    else:p=predict(trainer,ds)
                    violations=workstation_violations(p,settings["rounding_tolerance_hours"])
                    for j,target in enumerate(TARGETS):
                        y=ds.y.numpy()[:,j]
                        metrics.append({"machine":machine,"completeness":1.,"method":method,"seed":seed,"split":phase,"target":target,
                            **regression_metrics(y,p[:,j]),"constraint_violation":violations["any_violation_fraction"],
                            "accounting_violation_fraction":violations["violation_fractions"][2],
                            "mean_accounting_excess_hours":violations["mean_violation_hours"][2],
                            "n_train":len(ds_train),"n_test":len(parts["test"])})
                        predictions.extend({"machine":machine,"method":method,"seed":seed,"split":phase,"target":target,
                                            "record_id":identifier,"observed":float(a),"predicted":float(b)}
                                           for identifier,a,b in zip(f.record_id,y,p[:,j]))
            print(f"Evaluated {machine}, seed {seed}",flush=True)
    m=pd.DataFrame(metrics);p=pd.DataFrame(predictions);h=pd.DataFrame(histories)
    m.to_csv(out/"metrics.csv",index=False);p.to_csv(out/"predictions.csv",index=False);h.to_csv(out/"training_history.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(out/"training_diagnostics.csv",index=False)
    test=m[m.split=="test"]
    a=test.groupby(["machine","method","target"]).agg(MAE_mean=("MAE","mean"),MAE_sd=("MAE","std"),
                 RMSE_mean=("RMSE","mean"),R2_mean=("R2","mean"),WAPE_mean=("WAPE","mean"),
                 violation_mean=("constraint_violation","mean")).reset_index()
    a.to_csv(out/"metrics_aggregated.csv",index=False)
    elapsed=a[a.target=="elapsed_hours"]
    elapsed.to_latex(out/"paper_table.tex",index=False,escape=True,float_format="%.3f",caption="Synthetic diagnostic, not industrial validation. Seed SD concerns optimization only.")
    paired=test[test.method.isin(["data_driven","domain_informed"])].pivot(index=["machine","target","seed"],columns="method",values="MAE")
    paired["relative_mae_improvement"]=(paired.data_driven-paired.domain_informed)/paired.data_driven
    paired.to_csv(out/"paired_comparison.csv")
    plot_synthetic(frame,m,p,h,out/"figures")
    assert sha256(out/"selected_configs.json")==selection_hash
    assert all(sha256(path)==digest for path,digest in before.items())
    save_json(out/"validation.json",{"generator_repeat_exact":True,"first_seed_neural_retraining_exact":True,
              "finite_predictions":bool(np.isfinite(p[["observed","predicted"]]).all().all()),
              "industrial_raw_and_ms2_unchanged":True,"selected_config_hash":selection_hash,"test_used_for_tuning":False})
    sources=list((root/"src").rglob("*.py"))+list((root/"tests").glob("*.py"))+[config_path]
    save_json(out/"source_hashes.json",{str(path.relative_to(root)):sha256(path) for path in sorted(sources)})
    write_report(out,m,a,settings)
    print(elapsed.to_string(index=False),flush=True)
    return a


def write_report(out,metrics,aggregate,settings):
    elapsed=aggregate[aggregate.target=="elapsed_hours"]
    lines=["# Synthetic workstation diagnostic", "", "This is a controlled learnability check, not a reproduction of Company 2 and not MS3.","",
           "## Design", "", "Three single-server machines process 600 synthetic batch jobs each. Each has 360 training, "
           "120 validation and 120 test jobs, chronologically disjoint. Rates are 40, 65 and 90 units/hour. Product work, "
           "planned quantity, a small calendar effect and batch nonlinearity determine mean processing time. "
           "Independent 5% lognormal variation and positive pauses make the labels noisy. A FCFS arrival/start/completion "
           "recurrence supplies plausible serial event sequences. No physical capacity constraint is added to the neural model.","",
           "The neural architecture, loss, trainer and workstation constraints are reused. The only extra input in the "
           "full-feature models is planned quantity, standardized on training data. The limited-feature neural baseline "
           "uses the same product/operation/calendar feature types as the industrial run, without quantity. "
           "It is separately validation-tuned to avoid disadvantaging that baseline. The full-feature constrained and "
           "unconstrained models share architecture, optimizer settings, initial seed and minibatch ordering. "
           "Early-stopping epochs may differ. All configurations are frozen before evaluating the test jobs.","",
           "The generator's conditional mean is an audit-only reference, not a learned method or a literal lower bound "
           "on finite-sample MAE. The network never receives oracle values, hidden rates, realized pauses or queue states.","",
           "## Held-out elapsed-time results", "", "| Machine | Method | MAE h (mean ± seed SD) | R2 | WAPE | Any violation |",
           "|---|---|---:|---:|---:|---:|"]
    for r in elapsed.itertuples():
        lines.append(f"| {r.machine} | {r.method} | {r.MAE_mean:.3f} ± {r.MAE_sd:.3f} | {r.R2_mean:.3f} | {r.WAPE_mean:.1%} | {r.violation_mean:.1%} |")
    lines.extend(["","## Interpretation", "", "Learnability criterion fixed before evaluation: test R2 >= "
                  f"{settings['success']['minimum_test_r2']} and WAPE <= {settings['success']['maximum_test_wape']:.0%} "
                  "for every evaluation seed. This is separate from a claim that constraints improve accuracy."])
    for machine in sorted(metrics.machine.unique()):
        g=metrics[(metrics.machine==machine)&(metrics.split=="test")&(metrics.target=="elapsed_hours")]
        means=g.groupby("method").MAE.mean()
        good=g[g.method=="domain_informed"]
        passed=bool((good.R2.ge(settings["success"]["minimum_test_r2"])&good.WAPE.le(settings["success"]["maximum_test_wape"])).all())
        effect=(means.data_driven-means.domain_informed)/means.data_driven
        input_effect=(means.limited_features-means.data_driven)/means.limited_features
        lines.append(f"- {machine}: domain-informed learnability criterion {'met' if passed else 'not met'}. "
                     f"Providing quantity reduces unconstrained MAE by {input_effect:.1%}; adding constraints to full features "
                     f"changes MAE by {effect:+.1%} improvement (negative means worse).")
    lines.extend(["", "The full-feature versus limited-feature comparison tests the value of a known independent process driver. "
                  "The constrained versus unconstrained comparison tests the added value of soft feasibility penalties. "
                  "Good full-feature prediction alone does not establish that domain constraints caused it. "
                  "If unconstrained predictions already satisfy the broad inequalities, those inequalities provide little extra information.","",
                  "## Scientific limits", "", "This simulated mapping was deliberately made identifiable from the full input set. "
                  "Its clean job grain, observed planned quantity and stable product effects are stronger assumptions than the "
                  "industrial booking export supports. Do not transfer the synthetic accuracy to Company 2 or reinterpret "
                  "retrospective quantity confirmations as planned batch quantities. One simulation seed and three optimization "
                  "seeds are used: the SD is not uncertainty across simulation worlds. No missingness experiment, transport, "
                  "setup, inspection, failures or real-calendar shifts are modeled. Simulated queue waits are audit-only. "
                  "The experiment does not validate learned queue dynamics or single-server feasibility of predicted completions.","",
                  "## Validation and reproduction", "", "The generator reproduces exactly. Both full-feature neural methods "
                  "were trained twice at the first evaluation seed on each machine and required identical histories and parameters. "
                  "Raw Company 2 inputs and all existing MS2 result bytes were checked unchanged. "
                  "See validation.json, source_hashes.json, selected_configs.json and tuning_trials.csv.","",
                  "```powershell", "$env:PYTHONPATH = \"$PWD/src\"", "python -m pytest -q --basetemp=.pytest_tmp",
                  "python -m linn.experiments.workstation_synthetic --config configs/workstation_synthetic.json", "```", "",
                  "Outputs in results/synthetic/: simulated_bookings.csv, simulation_summary.csv, metrics.csv, "
                  "metrics_aggregated.csv, paired_comparison.csv, predictions.csv, training_history.csv, "
                  "training_diagnostics.csv, models/, encoder JSONs, figures/ and paper_table.tex. "
                  "Use synthetic_comparison and the per-machine reconstruction panels together. Industrial MS3 remains unstarted."])
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/workstation_synthetic.json")
    run(parser.parse_args().config)
