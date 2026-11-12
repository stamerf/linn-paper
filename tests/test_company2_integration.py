"""Read-only real-data integration checks; skipped only if private source data are absent."""
import json
from pathlib import Path
import pandas as pd
import pytest
import torch
from linn.data.loaders import load_company2
from linn.data.preprocessing import preprocess, machine_metadata, select_machines, chronological_split
from linn.experiments.workstation_ms1 import smoke


@pytest.mark.skipif(not Path("daten/manifest.json").exists(), reason="Company 2 private export unavailable")
def test_real_data_end_to_end():
    config = json.loads(Path("configs/workstation.json").read_text())
    tables, _, _ = load_company2("daten")
    clean,audit = preprocess(tables,config)
    selected, _ = select_machines(machine_metadata(clean,audit),config["selection"])
    assert selected == ["PSK 26","PSK 25","PSK 28"]
    split = chronological_split(clean,config["validation_start"],config["test_start"])
    assert split.groupby("order_id")["split"].nunique().max() == 1
    assert split.loc[split["split"].eq("train"),"end"].max() < pd.Timestamp(config["validation_start"])
    for machine in selected:
        train = split[split.machine.eq(machine)&split["split"].eq("train")]
        a,p,_,_ = smoke(train,config); b,q,_,_ = smoke(train,config)
        assert a == b and torch.equal(p,q)
