import inspect
import numpy as np
import pandas as pd
import torch
import pytest
from pathlib import Path
from linn.evaluation.metrics import regression_metrics
from linn.training.tuning import train_candidate,tune,predict
from linn.data.workstation_dataset import FeatureEncoder,WorkstationDataset
from linn.processes.workstation import workstation_spec
from linn.experiments.workstation_baseline import audit_relations


def test_validation_only_tuning_and_restored_checkpoint_reproducible():
    f=pd.DataFrame({"record_id":[str(i) for i in range(12)],"product":["A","B"]*6,
                    "operation":["10"]*12,"start":pd.date_range("2026-01-01",periods=12),
                    "elapsed_hours":np.linspace(1,2,12),"booked_hours":np.linspace(.8,1.8,12),"split":["train"]*12})
    train=f.iloc[:8].copy();val=f.iloc[8:].copy();val["split"]="validation"
    encoder=FeatureEncoder().fit(train);spec=workstation_spec(train)
    a,b=WorkstationDataset(train,encoder),WorkstationDataset(val,encoder)
    candidate={"hidden_dims":[4],"learning_rate":.01,"weight_decay":.001,"batch_size":4}
    settings={"max_epochs":5,"patience":2,"min_delta":.0001,"candidates":[candidate],
              "tuning_seeds":[1],"constraint_weights":[.1]}
    first,h,info=train_candidate(a,b,spec,candidate,.1,1,settings)
    second,h2,info2=train_candidate(a,b,spec,candidate,.1,1,settings)
    assert h==h2 and info==info2
    np.testing.assert_array_equal(predict(first,b),predict(second,b))
    actual=float(np.abs((predict(first,b)-b.y.numpy())/np.array(spec.context["target_scale"])).mean())
    assert abs(actual-info["validation_score"])<1e-8
    choice,trials=tune(a,b,spec,settings)
    assert len(trials)==2 and choice["constraint_weight"]==.1
    assert "test" not in inspect.signature(tune).parameters
    assert encoder.fitted_ids==train.record_id.tolist()


def test_commercial_relation_audit_does_not_silently_claim_independence():
    rel=pd.DataFrame({"source_bab_typ":["20"],"source_bab_nr":["1"],"linked_bab_typ":["20"],"linked_bab_nr":["2"]})
    f=pd.DataFrame({"order_id":["20:1","20:2"],"split":["train","test"]})
    audit=audit_relations(rel,f)
    assert audit["cross_split_relationship_pairs"]==1
    assert "not claimed" in audit["interpretation"]


@pytest.mark.skipif(not Path("results/ms2/predictions.csv").exists(),reason="MS2 experiment artifacts not present")
def test_saved_ms2_metrics_match_predictions_and_splits():
    p=pd.read_csv("results/ms2/predictions.csv",dtype={"record_id":str})
    metrics=pd.read_csv("results/ms2/metrics.csv")
    splits=pd.read_csv("results/ms2/split_manifest.csv",dtype={"record_id":str})
    assert splits.groupby("order_id")["split"].nunique().max()==1
    assert p.groupby(["machine","record_id","target"])["observed"].nunique().max()==1
    assert np.isfinite(p[["observed","predicted"]]).all().all()
    assert set(p.method)=={"median","data_driven","domain_informed"}
    assert p.seed.nunique()==3 and p.machine.nunique()==3
    for key,g in p.groupby(["machine","method","seed","split","target"]):
        row=metrics
        for col,value in zip(["machine","method","seed","split","target"],key): row=row[row[col]==value]
        assert len(row)==1
        recomputed=regression_metrics(g.observed,g.predicted)
        for name in ["MAE","RMSE","R2"]: assert row.iloc[0][name]==pytest.approx(recomputed[name])
