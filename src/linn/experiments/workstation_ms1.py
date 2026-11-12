"""End-to-end MS1 preparation and deterministic training smoke checks, not MS2 fitting."""
import argparse
import importlib.metadata
import importlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from linn.data.loaders import load_company2, sha256
from linn.data.preprocessing import preprocess, machine_metadata, select_machines, chronological_split
from linn.data.workstation_dataset import FeatureEncoder, WorkstationDataset
from linn.models.neural_model import NeuralRegressor
from linn.processes.workstation import workstation_spec
from linn.constraints.base_constraint import NoConstraints
from linn.training.trainer import Trainer, seed_everything
from linn.evaluation.plots import exploratory_plots


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def smoke(train, config, constrained=True):
    seed_everything(config["seed"])
    encoder = FeatureEncoder().fit(train)
    dataset = WorkstationDataset(train, encoder)
    spec = workstation_spec(train, config["rounding_tolerance_hours"])
    model = NeuralRegressor(dataset.x.shape[1], len(spec.target_names), **config["model"])
    trainer = Trainer(model, spec.constraints if constrained else NoConstraints(), spec.context,
                      config["training"]["learning_rate"], config["training"]["constraint_weight"])
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]), num_workers=0)
    initial = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    history = []
    for step, batch in enumerate(loader):
        history.append(trainer.step(batch))
        if step+1 >= config["training"]["smoke_steps"]:
            break
    final = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    if len(history) != config["training"]["smoke_steps"] or torch.equal(initial, final):
        raise AssertionError("Smoke check did not perform the requested parameter updates")
    with torch.no_grad():
        prediction = trainer.prediction(dataset.x)
    if not torch.isfinite(prediction).all():
        raise AssertionError("Nonfinite smoke prediction")
    return history, final, encoder.metadata(), spec.context


def run(config_path):
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (root / config["output_dir"]).resolve()
    data_dir = (root / config["data_dir"]).resolve()
    if output == data_dir or data_dir in output.parents:
        raise ValueError("Output must not be within the raw-data directory")
    output.mkdir(parents=True, exist_ok=True)
    tables, manifest, inventory = load_company2(data_dir)
    clean, audit = preprocess(tables, config)
    selected, ranking = select_machines(machine_metadata(clean, audit), config["selection"])
    # Split globally before selecting machines: an order's bookings on other machines also count.
    split = chronological_split(clean, config["validation_start"], config["test_start"])
    selected_frame = split[split.machine.isin(selected)].copy()
    safe_columns = ["record_id", "afo_lfd_nr", "machine", "order_id", "product", "operation", "start", "end",
                    "elapsed_hours", "booked_hours", "accounting_violation", "split"]
    selected_frame[safe_columns].to_csv(output / "prepared_bookings.csv", index=False)
    ranking.to_csv(output / "machine_selection.csv", index=False)
    full_meta = machine_metadata(clean, audit, development_only=False)
    full_meta[full_meta.machine.isin(selected)].to_csv(output / "selected_machine_metadata.csv", index=False)
    flags = ["missing_timestamp", "invalid_order", "zero_interval", "extreme_interval", "invalid_booked",
             "allocation_unsupported", "join_inconsistent", "accounting_violation"]
    audit[["record_id", "machine", "in_period", "development", "processing", "valid", "exclusion_reason"]+flags].to_csv(output / "quality_audit.csv", index=False)
    quality = {"concrete_resource_rows": len(audit), "processing_in_period": int((audit.processing & audit.in_period).sum()),
               "valid_all_machines": len(clean),
               "flags_processing_in_period": {c: int(audit.loc[audit.processing & audit.in_period,c].sum()) for c in flags}}
    save_json(output / "quality_summary.json", quality)
    save_json(output / "raw_inventory.json", inventory)
    save_json(output / "config.json", config)
    save_json(output / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {p: importlib.import_module(m).__version__ for p,m in
                           [("numpy","numpy"),("pandas","pandas"),("torch","torch"),
                            ("scikit-learn","sklearn"),("matplotlib","matplotlib"),("pytest","pytest")]},
              "distribution_metadata": {p: importlib.metadata.version(p) for p in ["numpy","pandas","torch","scikit-learn","matplotlib","pytest"]}})
    sources = list((root / "src").rglob("*.py")) + list((root / "tests").glob("*.py")) + [config_path]
    sources += list((root / "data analysis").glob("*")) + list((root / "docs").glob("*.md"))
    sources += list((root / "notebooks").glob("*.ipynb"))
    sources += [root / "README.md", root / "pyproject.toml", root / "requirements-ms1.txt"]
    save_json(output / "source_hashes.json", {str(p.relative_to(root)): sha256(p) for p in sorted(sources) if p.is_file()})
    split_counts = selected_frame.groupby(["machine", "split"]).size().unstack(fill_value=0)
    split_counts.to_csv(output / "split_counts.csv")
    checks, histories = {}, []
    for machine in selected:
        train = selected_frame[selected_frame.machine.eq(machine) & selected_frame["split"].eq("train")]
        for name in ["train", "validation", "test"]:
            if not (selected_frame.machine.eq(machine) & selected_frame["split"].eq(name)).any():
                raise AssertionError(f"Empty {name} split for {machine}")
        first, weights1, encoding, context = smoke(train, config)
        second, weights2, _, _ = smoke(train, config)
        baseline, _, _, _ = smoke(train, config, constrained=False)
        if first != second or not torch.equal(weights1, weights2):
            raise AssertionError("Repeated seeded CPU smoke runs differ")
        checks[machine] = {"deterministic_exact": True, "finite": True, "steps_per_run": len(first),
                           "train_records": len(train), "input_dim": len(encoding["feature_names"]),
                           "context": context, "constraint_names": list(first[0].keys())[3:]}
        save_json(output / (machine.replace(" ", "_")+"_encoder.json"), encoding)
        for method, history in [("domain_informed", first), ("data_driven", baseline)]:
            histories.extend({"machine": machine, "method": method, "step": i+1, **h} for i,h in enumerate(history))
    pd.DataFrame(histories).to_csv(output / "smoke_history.csv", index=False)
    save_json(output / "smoke_checks.json", checks)
    exploratory_plots(selected_frame, full_meta, output / "figures")
    # Confirm all source data bytes stayed unchanged.
    for name, entry in inventory.items():
        if isinstance(entry, dict) and sha256(data_dir / entry["filename"]) != entry["sha256"]:
            raise AssertionError("Raw data changed during run")
    if sha256(data_dir / "manifest.json") != inventory["manifest_sha256"]:
        raise AssertionError("Manifest changed during run")
    lines = ["# MS1 machine selection and validation", "", "MS1 only: no converged models or held-out predictive scores are claimed.", "",
             "## Reproducible selection", "", "Selection uses only processing bookings starting before 2026-06-01. "
             "Concrete machine definition follows notebook cell 31. Eligible APS die-casting presses need "
             "100 records, 15 orders, 8 products, 95% valid records and nonzero duration variation. "
             "Rank = log(1 + valid records) × valid fraction × (1 − overlap fraction). "
             "Take the best three distinct machine descriptions; tie-break by machine ID. "
             "The description rule favors different nominal press capacities. Full rankings include rejected machines.", "",
             "| Machine | Description | Development bookings | Orders | Products | Overlap | Score |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for machine in selected:
        r = ranking.set_index("machine").loc[machine]
        lines.append(f"| {machine} | {r.description} | {r.n_records} | {r.n_orders} | {r.n_products} | {r.overlap_fraction:.1%} | {r.score:.3f} |")
    lines.extend(["", "## Prepared cohort and split", "", "```text", split_counts.to_string(), "```", "",
                  "Boundaries: training before June 1; validation June 1–July 15; test from July 16. "
                  "Whole production orders crossing boundaries are purged globally across all concrete machines. "
                  "No validation/test prediction metrics are computed in MS1. Selection is conditioned on the early cohort; "
                  "full-period figures are descriptive audits only.", "", "## Validation", "",
                  "Each selected machine passed two exactly matching seeded CPU runs of three forward/backward optimizer steps, "
                  "plus a no-constraint smoke run using the identical neural architecture. Losses, gradients, parameters, "
                  "and outputs were finite. Raw hashes were unchanged. See smoke_checks.json and smoke_history.csv. "
                  "Automated test status is recorded separately in validation.txt.", "", "## Scope and assumptions", "",
                  "Targets are elapsed processing-booking hours and booked processing hours, not individual part-cycle time. "
                  "Quantity confirmations are separate records. Setup/mixed records and the explicit disturbance article are excluded. "
                  "Intervals over 24 hours are quarantined as a stated short-booking cohort restriction, not silently clipped. "
                  "Accounting violations are retained and flagged. Overlap is diagnostic; no single-server capacity, queue, "
                  "flow conservation, or utilization constraint is asserted. See docs/ms1_data_and_model.md for provenance and limitations.", "",
                  "Reproduce from the repository root in PowerShell:", "", "```powershell",
                  "$env:PYTHONPATH = \"$PWD/src\"", "python -m pytest -q --basetemp=.pytest_tmp",
                  "python -m linn.experiments.workstation_ms1 --config configs/workstation.json", "```"])
    (output.parent / "ms1_machine_selection.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"milestone": 1, "selected_machines": selected, "smoke_checks": "passed", "output": str(output)}, indent=2))
    return selected_frame, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/workstation.json")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
