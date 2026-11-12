import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from linn.data.loaders import load_company2
from linn.data.preprocessing import parse_wall_time, preprocess, unique_records, chronological_split, select_machines
from linn.data.workstation_dataset import FeatureEncoder, WorkstationDataset
from linn.constraints.base_constraint import ConstraintSet, NoConstraints
from linn.constraints.workstation_constraints import WorkstationConstraints
from linn.models.neural_model import NeuralRegressor
from linn.training.losses import masked_mse
from linn.training.trainer import Trainer, seed_everything
from linn.evaluation.metrics import regression_metrics, workstation_violations


@pytest.fixture
def config():
    return json.loads(Path("configs/workstation.json").read_text())


def toy_tables():
    afo = pd.DataFrame([dict(afo_lfd_nr="1", afo_art_nr="P1", afo_tr="0", afo_teg="1", afo_tg="1",
                            afo_bab_typ="20", afo_bab_nr="001", afo_bab_afo_nr="10", bab_belegart=None)])
    resource = pd.DataFrame([dict(afo_lfd_nr="1", res_lfd_nr="001", kst_typ="10", kst_roeders_kstgrp="APS",
                                 res_kst_id="PSK 01", res_beginn="2026-01-05T10:00", res_ende="2026-01-05T11:00:00.123",
                                 afo_datum="2026-01-05T11:00", res_zeit="1", res_zeitanteil="100",
                                 afo_bab_typ="20", afo_bab_nr="001", afo_bab_afo_nr="10", kst_bez1="Press")])
    return {"afoRows": afo, "resourceRows": resource}


def test_mixed_timestamp_precision_and_invalid():
    s = parse_wall_time(pd.Series(["2026-01-01T10:00", "2026-01-01T10:00:01.123", "bad", None]))
    assert s.notna().tolist() == [True, True, False, False]
    assert s.iloc[1].microsecond == 123000


def test_preprocessing_and_duplicate_policy(config):
    tables = toy_tables()
    tables["resourceRows"] = pd.concat([tables["resourceRows"]]*2, ignore_index=True)
    clean, audit = preprocess(tables, config)
    assert len(clean) == 1 and clean.record_id.iloc[0] == "001"
    assert clean.order_id.iloc[0] == "20:001"
    assert clean.elapsed_hours.iloc[0] > 1
    tables["resourceRows"].loc[1,"res_zeit"] = "2"
    with pytest.raises(ValueError, match="conflicting"):
        preprocess(tables, config)


@pytest.mark.parametrize("column,value,flag", [
    ("res_beginn", None, "missing_timestamp"), ("res_ende", "bad", "missing_timestamp"),
    ("res_ende", "2026-01-05T09:00", "invalid_order"),
    ("res_ende", "2026-01-05T10:00", "zero_interval"),
    ("res_ende", "2026-01-07T10:00", "extreme_interval"),
    ("res_zeit", "-1", "invalid_booked"), ("res_zeit", "inf", "invalid_booked"),
    ("res_zeitanteil", "101", "allocation_unsupported"),
    ("afo_bab_nr", "other", "join_inconsistent")])
def test_quality_flags(config, column, value, flag):
    tables = toy_tables()
    second = tables["resourceRows"].copy()
    second["res_lfd_nr"] = "002"
    second[column] = value
    tables["resourceRows"] = pd.concat([tables["resourceRows"], second], ignore_index=True)
    clean, audit = preprocess(tables, config)
    assert len(clean) == 1
    assert bool(audit.loc[audit.record_id.eq("002"),flag].iloc[0])


def test_accounting_violation_is_retained(config):
    tables = toy_tables()
    tables["resourceRows"]["res_zeit"] = "2"
    clean, _ = preprocess(tables, config)
    assert len(clean) == 1 and clean.accounting_violation.iloc[0]


@pytest.mark.parametrize("col,value", [("afo_tr","1"), ("afo_teg","0"), ("afo_art_nr","#DG-STÖRZEITEN")])
def test_nonprocessing_excluded(config, col, value):
    tables = toy_tables()
    tables["afoRows"][col] = value
    with pytest.raises(ValueError, match="No valid"):
        preprocess(tables, config)


def test_loader_manifest_and_string_ids(tmp_path):
    manifest = {"entries": {"afoRows":"a.csv"}, "counts":{"afoRows":1}}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    (tmp_path/"a.csv").write_text("id,attribute\n001,NA\n")
    tables, _, inventory = load_company2(tmp_path)
    assert tables["afoRows"].iloc[0].tolist() == ["001","NA"]
    assert len(inventory["afoRows"]["sha256"]) == 64
    manifest["counts"]["afoRows"] = 2
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        load_company2(tmp_path)


def test_split_purges_orders_and_straddling_intervals():
    f = pd.DataFrame({"order_id":["a","b","b","c","d","e"],
                      "start":pd.to_datetime(["2026-01-01","2026-05-31","2026-06-02","2026-06-15","2026-07-20","2026-05-31"]),
                      "end":pd.to_datetime(["2026-01-02","2026-05-31","2026-06-03","2026-06-16","2026-07-21","2026-06-01"])})
    result = chronological_split(f,"2026-06-01","2026-07-16")
    assert result["split"].tolist() == ["train","purged","purged","validation","test","purged"]
    assert result.groupby("order_id")["split"].nunique().max() == 1
    with pytest.raises(ValueError):
        chronological_split(f,"2026-07-16","2026-06-01")


def sample_frame():
    return pd.DataFrame({"record_id":["1","2","3"], "product":["A","B","A"], "operation":["10"]*3,
                         "start":pd.to_datetime(["2026-01-01","2026-01-02","2026-01-03"]),
                         "elapsed_hours":[1.,2.,3.], "booked_hours":[.9,1.8,2.7], "split":["train"]*3})


def test_train_only_encoding_unknown_category_and_no_target_leakage():
    train = sample_frame()
    encoder = FeatureEncoder().fit(train)
    other = train.copy(); other["product"] = "unseen"; other["elapsed_hours"] = 999
    assert np.isfinite(encoder.transform(other)).all()
    modified = train.copy(); modified["booked_hours"] = -999; modified["elapsed_hours"] = 999
    np.testing.assert_array_equal(encoder.transform(train),encoder.transform(modified))
    other["split"] = "test"
    with pytest.raises(ValueError, match="training"):
        FeatureEncoder().fit(other)


def test_masked_targets_cannot_affect_loss_or_gradient():
    p = torch.tensor([[1.,2.]],requires_grad=True)
    mask = torch.tensor([[True,False]])
    loss = masked_mse(p,torch.tensor([[3.,float("nan")]]),mask,[1,1])
    loss.backward()
    assert loss.item() == 4 and p.grad.tolist() == [[-4.,0.]]
    assert masked_mse(p,torch.full_like(p,float("nan")),torch.zeros_like(mask),[1,1]).item() == 0
    with pytest.raises(ValueError):
        masked_mse(p,torch.ones(2),mask,[1,1])
    frame = sample_frame(); frame["booked_hours"] = np.nan
    ds = WorkstationDataset(frame,FeatureEncoder().fit(frame))
    assert not ds.mask[:,1].any() and torch.isfinite(ds.y).all()


def test_constraint_values_and_directional_gradients():
    p = torch.tensor([[-1.,2.]],requires_grad=True)
    terms = WorkstationConstraints().compute(p,{}, {"constraint_scale":1.,"rounding_tolerance_hours":0.})
    assert {k:v.item() for k,v in terms.items()} == {"elapsed_nonnegative":1.,"booked_nonnegative":0.,"booked_within_elapsed":9.}
    sum(terms.values()).backward()
    assert p.grad[0,0] < 0 and p.grad[0,1] > 0
    valid = WorkstationConstraints().compute(torch.tensor([[2.,1.]]),{}, {"constraint_scale":1.,"rounding_tolerance_hours":.02})
    assert all(v.item() == 0 for v in valid.values())
    with pytest.raises(ValueError):
        WorkstationConstraints().compute(torch.ones(3,1),{}, {})


def test_generic_trainer_supports_unrelated_constraint_and_reproducibility():
    class DifferentProcessConstraint(ConstraintSet):
        def compute(self,prediction,batch,context):
            assert "targets" not in batch and "target_mask" not in batch
            return {"example_ceiling":torch.relu(prediction-10).square().mean()}
    def run():
        seed_everything(42)
        model = NeuralRegressor(4,3,(8,))
        trainer = Trainer(model,DifferentProcessConstraint(),{"target_scale":[1,1,1]})
        batch = {"features":torch.randn(5,4),"targets":torch.ones(5,3),"target_mask":torch.ones(5,3,dtype=torch.bool)}
        before = model(batch["features"]).detach().clone()
        history = trainer.step(batch)
        after = model(batch["features"]).detach()
        assert not torch.equal(before,after)
        assert np.isfinite(list(history.values())).all()
        return history,after
    a,p = run(); b,q = run()
    assert a == b and torch.equal(p,q)
    assert NoConstraints().compute(p,{}, {}) == {}


def test_metrics_hand_computed_and_undefined_cases():
    m = regression_metrics([1,2,3],[2,2,2])
    assert m["MAE"] == pytest.approx(2/3) and m["R2"] == 0
    assert regression_metrics([0,0],[0,0])["R2"] is None
    assert regression_metrics([0,0],[0,0])["WAPE"] is None
    assert workstation_violations([[1,2]])["any_violation_fraction"] == 1
    with pytest.raises(ValueError):
        regression_metrics([1],[np.nan])


def test_selection_independent_of_row_order(config):
    rows = [dict(machine=str(i),description=str(i),group="APS",n_records=150+i,n_orders=20,n_products=10,
                 valid_fraction=1.,overlap_fraction=.1,elapsed_std=1.) for i in range(4)]
    frame = pd.DataFrame(rows)
    first, _ = select_machines(frame,config["selection"])
    second, _ = select_machines(frame.sample(frac=1,random_state=1),config["selection"])
    assert first == second == ["3","2","1"]
