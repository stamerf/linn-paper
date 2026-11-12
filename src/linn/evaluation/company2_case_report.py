"""Descriptive reports for the exploratory Company 2 aggregate-booking case.

Seeds repeat masks/optimization on the same industrial holdout. They are not
independent factories, orders, or confidence-interval replications.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import regression_metrics, workstation_violations


METHODS = ["median", "quantity_linear", "data_driven", "domain_informed", "without_quantity"]
LABELS = {"median": "Training median", "quantity_linear": "Quantity-aware ridge",
          "data_driven": "Data-driven NN", "domain_informed": "Domain-informed NN",
          "without_quantity": "NN without quantity"}
COLORS = {"median": "#777777", "quantity_linear": "#009E73",
          "data_driven": "#0072B2", "domain_informed": "#D55E00", "without_quantity": "#CC79A7"}
TARGETS = {"elapsed_hours": "Summed booking-window hours",
           "booked_hours": "Summed booked hours"}


def _aggregate(frame, keys):
    specification = {}
    for metric in ["MAE", "RMSE", "R2", "WAPE", "constraint_violation"]:
        if metric in frame:
            specification[metric + "_mean"] = (metric, "mean")
            specification[metric + "_seed_sd"] = (metric, "std")
    specification["n_seeds"] = ("seed", "nunique")
    for count in ["n_train", "n_test", "n_observed_train_per_target",
                  "n_observed_validation_per_target", "n_records", "n_orders"]:
        if count in frame:
            specification[count + "_min"] = (count, "min")
            specification[count + "_max"] = (count, "max")
    return frame.groupby(keys, dropna=False).agg(**specification).reset_index()


def _per_machine(predictions, tolerance):
    rows = []
    for keys, group in predictions[predictions.split.eq("test")].groupby(
            ["machine", "completeness", "method", "seed"], sort=True):
        machine, completeness, method, seed = keys
        prediction = group[["elapsed_predicted", "booked_predicted"]].to_numpy()
        violations = workstation_violations(prediction, tolerance)
        for target, prefix in [("elapsed_hours", "elapsed"), ("booked_hours", "booked")]:
            rows.append({"machine": machine, "completeness": completeness,
                         "method": method, "seed": seed, "target": target,
                         "n_records": len(group), "n_orders": group.order_id.nunique(),
                         "constraint_violation": violations["any_violation_fraction"],
                         **regression_metrics(group[prefix + "_observed"],
                                              group[prefix + "_predicted"])})
    return pd.DataFrame(rows)


def _convergence(out, metrics, settings):
    history_path = out / "history.csv"
    if not history_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    history = pd.read_csv(history_path)
    keys = ["completeness", "method", "seed"]
    rows = []
    for values, group in history.groupby(keys, sort=True):
        group = group.sort_values("epoch")
        completeness, method, seed = values
        info = metrics[metrics.completeness.eq(completeness) & metrics.method.eq(method) &
                       metrics.seed.eq(seed)]
        if info.empty or "best_epoch" not in info or info.best_epoch.dropna().empty:
            raise ValueError("Training history requires saved best-epoch metadata")
        best_epoch = int(info.best_epoch.dropna().iloc[0])
        chosen = group[group.epoch.eq(best_epoch)]
        if len(chosen) != 1 or group.epoch.duplicated().any():
            raise ValueError("Missing or ambiguous restored epoch in training history")
        best, last = chosen.iloc[0], group.iloc[-1]
        numeric = group.select_dtypes(include=[np.number])
        if not np.isfinite(numeric).all().all():
            raise ValueError("Nonfinite training-history diagnostic")
        n_epochs = len(group)
        if "epochs_run" in info and int(info.epochs_run.dropna().iloc[0]) != n_epochs:
            raise ValueError("Training-history length differs from run metadata")
        row = dict(zip(keys, values))
        row.update(best_epoch=best_epoch, epochs_run=n_epochs,
                   epoch_budget_reached=n_epochs >= int(settings.get("max_epochs", n_epochs + 1)),
                   train_normalized_mae_at_best=float(best.train_normalized_mae),
                   validation_normalized_mae_at_best=float(best.validation_normalized_mae),
                   validation_normalized_mae_at_end=float(last.validation_normalized_mae),
                   train_normalized_mae_at_end=float(last.train_normalized_mae),
                   validation_minus_train_at_best=float(best.validation_normalized_mae-best.train_normalized_mae))
        for name in ["train_constraint_loss", "validation_constraint_loss", "train_data_loss",
                     "validation_data_loss", "train_total_loss", "validation_total_loss"]:
            if name in group:
                row[name + "_at_best"] = float(best[name])
        rows.append(row)
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(out / "convergence_diagnostics.csv", index=False)
    return diagnostics, history


def summarize(out, settings):
    """Read final runner artifacts; write descriptive tables, plots and summary."""
    out = Path(out)
    metrics = pd.read_csv(out / "metrics.csv")
    predictions = pd.read_csv(out / "predictions.csv", dtype={"record_id": str, "order_id": str})
    cohort_path = out / "cohort_summary.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8")) if cohort_path.exists() else {}
    required = ["machine", "target", "split", "completeness", "method", "seed"]
    if metrics.duplicated(required).any():
        raise ValueError("Duplicate run/target metric rows")
    if not np.isfinite(metrics[["MAE", "RMSE", "constraint_violation"]]).all().all():
        raise ValueError("Nonfinite primary metrics")
    pcols = ["elapsed_observed", "booked_observed", "elapsed_predicted", "booked_predicted"]
    test_predictions = predictions[predictions.split.eq("test")]
    if test_predictions.empty or not np.isfinite(test_predictions[pcols]).all().all():
        raise ValueError("Complete finite test predictions are required")
    run_keys = ["split", "completeness", "method", "seed", "record_id"]
    if test_predictions.duplicated(run_keys).any():
        raise ValueError("Duplicate predictions within a run")
    reference = None
    for _, group in test_predictions.groupby(["completeness", "method", "seed"]):
        ordered = group.sort_values("record_id").set_index("record_id")
        observed = ordered[["order_id", "machine", "elapsed_observed", "booked_observed"]]
        if reference is None:
            reference = observed
        elif not reference.equals(observed):
            raise ValueError("Test reference differs between methods, masks, or seeds")

    aggregated = _aggregate(metrics, ["split", "machine", "target", "completeness", "method"])
    aggregated.to_csv(out / "metrics_aggregated.csv", index=False)
    neural = metrics[metrics.method.isin(["data_driven", "domain_informed"])]
    pairs = neural.pivot(index=["split", "machine", "target", "completeness", "seed"],
                         columns="method", values="MAE").reset_index()
    if pairs[["data_driven", "domain_informed"]].isna().any().any():
        raise ValueError("Each neural comparison requires both paired methods")
    pairs["absolute_improvement_hours"] = pairs.data_driven - pairs.domain_informed
    pairs["relative_improvement"] = np.divide(
        pairs.absolute_improvement_hours, pairs.data_driven,
        out=np.full(len(pairs), np.nan), where=pairs.data_driven.to_numpy() != 0)
    pairs.to_csv(out / "paired_comparison.csv", index=False)
    effects = pairs.groupby(["split", "machine", "target", "completeness"]).agg(
        improvement_mean=("relative_improvement", "mean"),
        improvement_seed_sd=("relative_improvement", "std"),
        minimum_seed_improvement=("relative_improvement", "min"),
        maximum_seed_improvement=("relative_improvement", "max"),
        improved_seeds=("absolute_improvement_hours", lambda x: int((x > 0).sum())),
        n_seeds=("seed", "nunique"),
        absolute_improvement_hours_mean=("absolute_improvement_hours", "mean")).reset_index()
    effects.to_csv(out / "effect_summary.csv", index=False)
    tolerance = float(settings.get("rounding_tolerance_hours", 0.02))
    machine_metrics = _per_machine(predictions, tolerance)
    machine_metrics.to_csv(out / "per_machine_metrics.csv", index=False)
    machine_aggregate = _aggregate(machine_metrics, ["machine", "target", "completeness", "method"])
    machine_aggregate.to_csv(out / "per_machine_metrics_aggregated.csv", index=False)
    prior = {str(name).replace(" ", "").lower() for name in
             settings.get("prior_test_machines", ["PSK 25", "PSK 26", "PSK 28"])}
    subsets = test_predictions.copy()
    subsets["machine"] = np.where(
        subsets.machine.astype(str).str.replace(" ", "", regex=False).str.lower().isin(prior),
        "previously_evaluated_machines", "previously_unevaluated_machines")
    subset_metrics = _per_machine(subsets, tolerance).rename(columns={"machine": "evaluation_subset"})
    subset_metrics.to_csv(out / "test_subset_metrics.csv", index=False)
    subset_aggregate = _aggregate(subset_metrics, ["evaluation_subset", "target", "completeness", "method"])
    subset_aggregate.to_csv(out / "test_subset_metrics_aggregated.csv", index=False)
    convergence, history = _convergence(out, metrics, settings)

    # Tables average metrics of individual fits. The scatter is explicitly a
    # median-prediction illustration, never substituted for those run metrics.
    test = aggregated[aggregated.split.eq("test") & aggregated.machine.eq("pooled")]
    elapsed = test[test.target.eq("elapsed_hours")]
    paper = elapsed[["completeness", "method", "MAE_mean", "MAE_seed_sd", "RMSE_mean",
                     "R2_mean", "WAPE_mean", "constraint_violation_mean"]].copy()
    paper["method"] = paper.method.map(LABELS)
    paper.to_latex(out / "paper_table.tex", index=False, escape=True, float_format="%.3f",
                   caption="Company 2 retrospective reconstruction of summed processing-booking windows. "
                   "MAE and RMSE are in hours. Seed SD describes masks and optimization on the same holdout; "
                   "it is not a confidence interval. Completeness concerns duration labels, with quantities retained.")
    _figures(out, test, effects, test_predictions, machine_aggregate)
    if not history.empty:
        _learning_curves(out, history, convergence, settings)
    _summary(out, settings, cohort, metrics, elapsed, effects, reference, tolerance, subset_aggregate, convergence)
    return {"metric_rows": len(metrics), "test_records": len(reference),
            "test_orders": int(reference.order_id.nunique()),
            "figures": ["case_overview", "predicted_observed", "per_machine_comparison"] +
                       (["learning_curves"] if not history.empty else [])}


def _figures(out, aggregate, effects, predictions, machines):
    folder = out / "figures"
    folder.mkdir(exist_ok=True)

    def save(figure, name):
        figure.tight_layout()
        for extension in ["png", "pdf"]:
            figure.savefig(folder / (name + "." + extension), dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for target, axis in zip(TARGETS, axes[0]):
        for method in METHODS:
            group = aggregate[aggregate.target.eq(target) & aggregate.method.eq(method)].sort_values("completeness")
            if group.empty:
                continue
            axis.errorbar(group.completeness.to_numpy() * 100, group.MAE_mean.to_numpy(),
                          yerr=group.MAE_seed_sd.fillna(0).to_numpy(), fmt="o-",
                          label=LABELS[method], color=COLORS[method], capsize=3)
        axis.set(title=TARGETS[target], ylabel="MAE (hours)")
        axis.legend(fontsize=8)
    effect = effects[effects.split.eq("test") & effects.machine.eq("pooled") &
                     effects.target.eq("elapsed_hours")].sort_values("completeness")
    axes[1, 0].errorbar(effect.completeness.to_numpy() * 100,
                       effect.improvement_mean.to_numpy() * 100,
                       yerr=effect.improvement_seed_sd.fillna(0).to_numpy() * 100,
                       fmt="o-", color="#009E73", capsize=3)
    axes[1, 0].axhline(0, color="black", lw=1)
    axes[1, 0].set(title="Paired effect: summed window hours", ylabel="Relative MAE improvement (%)")
    for method in METHODS:
        group = aggregate[aggregate.target.eq("elapsed_hours") & aggregate.method.eq(method)].sort_values("completeness")
        if group.empty:
            continue
        axes[1, 1].plot(group.completeness.to_numpy() * 100,
                        group.constraint_violation_mean.to_numpy() * 100,
                        "o-", label=LABELS[method], color=COLORS[method])
    axes[1, 1].set(title="Any predicted accounting violation", ylabel="Test aggregates (%)", ylim=(-2, 102))
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel("Training / validation duration labels retained (%)")
        axis.invert_xaxis()
        axis.grid(alpha=.18)
    figure.suptitle("Company 2: same chronological holdout; error bars = seed SD", fontsize=12)
    save(figure, "case_overview")

    levels = [float(predictions.completeness.max()), float(predictions.completeness.min())]
    levels = list(dict.fromkeys(levels))
    figure, axes = plt.subplots(len(levels), 2, figsize=(10, 4.4 * len(levels)), squeeze=False)
    for row, level in enumerate(levels):
        for col, (prefix, target) in enumerate([("elapsed", "elapsed_hours"), ("booked", "booked_hours")]):
            axis = axes[row, col]
            extrema = [0.]
            for method in METHODS:
                group = predictions[predictions.completeness.eq(level) & predictions.method.eq(method)]
                if group.empty:
                    continue
                medians = group.groupby("record_id")[[prefix + "_observed", prefix + "_predicted"]].median()
                observed = medians[prefix + "_observed"].to_numpy()
                predicted = medians[prefix + "_predicted"].to_numpy()
                axis.scatter(observed, predicted, s=20, alpha=.65, label=LABELS[method], color=COLORS[method])
                extrema.extend([observed.min(), observed.max(), predicted.min(), predicted.max()])
            lo, hi = min(extrema), max(extrema)
            padding = max((hi - lo) * .05, .1)
            axis.plot([lo, hi], [lo, hi], "k--", lw=1)
            axis.set(xlim=(lo-padding, hi+padding), ylim=(lo-padding, hi+padding),
                     title=f"{level:.0%} labels: {TARGETS[target]}",
                     xlabel="Observed aggregate hours", ylabel="Predicted hours (median across seeds)")
            axis.legend(fontsize=8)
            axis.grid(alpha=.18)
    figure.suptitle("Illustrative median-prediction ensembles; all held-out aggregates shown", fontsize=12)
    save(figure, "predicted_observed")

    figure, axes = plt.subplots(1, len(levels), figsize=(6 * len(levels), 4.6), squeeze=False)
    for axis, level in zip(axes.flat, levels):
        subset = machines[machines.target.eq("elapsed_hours") & machines.completeness.eq(level)]
        names = sorted(subset.machine.unique())
        x = np.arange(len(names))
        present = [method for method in METHODS if method in subset.method.unique()]
        width = .8 / max(len(present), 1)
        for index, method in enumerate(present):
            group = subset[subset.method.eq(method)].set_index("machine").reindex(names)
            axis.bar(x + (index - (len(present)-1)/2) * width, group.MAE_mean.to_numpy(), width,
                     yerr=group.MAE_seed_sd.fillna(0).to_numpy(), color=COLORS[method],
                     label=LABELS[method], capsize=2)
        axis.set_xticks(x)
        axis.set_xticklabels(names, rotation=25, ha="right")
        axis.set(title=f"{level:.0%} duration labels", ylabel="Summed window MAE (hours)")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=.18)
    figure.suptitle("Machine diagnostics within one pooled model; seed SD is descriptive", fontsize=12)
    save(figure, "per_machine_comparison")


def _learning_curves(out, history, convergence, settings):
    # Fixed representative optimizer seed, never chosen using held-out quality.
    seed = int(settings.get("representative_plot_seed", 101))
    present = history[history.seed.eq(seed)]
    if present.empty:
        raise ValueError(f"Preselected illustrative seed {seed} is missing")
    levels = [float(present.completeness.max()), float(present.completeness.min())]
    levels = list(dict.fromkeys(levels))
    figure, axes = plt.subplots(len(levels), 3, figsize=(14, 4 * len(levels)), squeeze=False)
    for row, level in enumerate(levels):
        for method in ["data_driven", "domain_informed"]:
            group = present[present.completeness.eq(level) & present.method.eq(method)].sort_values("epoch")
            if group.empty:
                continue
            selected = convergence[convergence.seed.eq(seed) & convergence.method.eq(method) &
                                   convergence.completeness.eq(level)].iloc[0]
            for column, name in enumerate(["train_normalized_mae", "validation_normalized_mae"]):
                axes[row, column].plot(group.epoch.to_numpy(), group[name].to_numpy(),
                                       color=COLORS[method], label=LABELS[method])
                best = group[group.epoch.eq(selected.best_epoch)].iloc[0]
                axes[row, column].scatter([selected.best_epoch], [best[name]],
                                          color=COLORS[method], marker="x", s=50)
            for split, style in [("train", "-"), ("validation", "--")]:
                name = split + "_constraint_loss"
                if name in group:
                    axes[row, 2].plot(group.epoch.to_numpy(), group[name].to_numpy(),
                                      linestyle=style, color=COLORS[method],
                                      label=LABELS[method] + " " + split)
        axes[row, 0].set(title=f"{level:.0%} labels: observed training error", ylabel="Normalized MAE")
        axes[row, 1].set(title=f"{level:.0%} labels: observed validation error", ylabel="Normalized MAE")
        axes[row, 2].set(title=f"{level:.0%} labels: constraint diagnostics", ylabel="Unweighted penalty sum")
        axes[row, 2].set_yscale("symlog", linthresh=1e-6)
        for axis in axes[row]:
            axis.set_xlabel("Epoch")
            axis.legend(fontsize=7)
            axis.grid(alpha=.18)
    figure.suptitle(f"Preselected illustrative seed {seed}; crosses mark restored checkpoints", fontsize=12)
    figure.tight_layout()
    for extension in ["png", "pdf"]:
        figure.savefig(out / "figures" / ("learning_curves." + extension), dpi=180)
    plt.close(figure)


def _summary(out, settings, cohort, metrics, elapsed, effects, reference, tolerance, subset_aggregate, convergence):
    levels = sorted(elapsed.completeness.unique(), reverse=True)
    lines = ["# Company 2: retrospective aggregate-booking reconstruction", "",
             "This is an exploratory industrial case using one pooled model for the selected APS presses. "
             "It reconstructs recorded processing accounting from independently retained confirmed quantities "
             "and categorical descriptors. It is not a prospective completion-time forecast or a part-cycle model.", "",
             "## Held-out results", "",
             f"The chronological test reference contains {len(reference)} order-operation-machine aggregates "
             f"from {reference.order_id.nunique()} production orders and {reference.machine.nunique()} presses. "
             "Every method, completeness level and seed is evaluated on the same reference. "
             "Pooled metrics weight aggregates equally, not machines equally.", "",
             "The primary elapsed target is the **sum of individual processing-booking window durations**, "
             "not the wall-clock span between an operation's first start and last finish. Booked hours are "
             "summed resource-time entries. Confirmed quantity is retrospective; retaining it supports "
             "reconstruction of missing time accounting, not a claim that quantity was known before production.", "",
             "| Duration labels retained | Method | MAE h (mean +/- seed SD) | RMSE h | R2 | WAPE | Any violation |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for level in levels:
        for method in METHODS:
            rows = elapsed[elapsed.completeness.eq(level) & elapsed.method.eq(method)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            deviation = row.MAE_seed_sd if pd.notna(row.MAE_seed_sd) else 0.
            lines.append(f"| {level:.0%} | {LABELS[method]} | {row.MAE_mean:.3f} +/- {deviation:.3f} | "
                         f"{row.RMSE_mean:.3f} | {row.R2_mean:.3f} | {row.WAPE_mean:.1%} | "
                         f"{row.constraint_violation_mean:.1%} |")
    lines += ["", "## Interpretation", ""]
    for level in levels:
        selected = effects[effects.split.eq("test") & effects.machine.eq("pooled") &
                           effects.target.eq("elapsed_hours") & effects.completeness.eq(level)]
        if selected.empty:
            continue
        row = selected.iloc[0]
        lines.append(f"- At {level:.0%} duration labels, the mean paired domain-informed MAE improvement "
                     f"is {row.improvement_mean:+.1%}; {int(row.improved_seeds)}/{int(row.n_seeds)} paired "
                     f"seeds improve (range {row.minimum_seed_improvement:+.1%} to "
                     f"{row.maximum_seed_improvement:+.1%}). Positive means lower error.")
    full = elapsed[elapsed.completeness.eq(max(levels))].set_index("method")
    if "domain_informed" in full.index:
        model = full.loc["domain_informed"]
        for baseline in ["median", "quantity_linear"]:
            if baseline in full.index and full.loc[baseline, "MAE_mean"] > 0:
                improvement = 1-model.MAE_mean/full.loc[baseline, "MAE_mean"]
                lines.append(f"- With the most complete labels, domain-informed MAE is "
                             f"{abs(improvement):.1%} {'lower' if improvement >= 0 else 'higher'} "
                             f"than {LABELS[baseline].lower()} (ratio of mean errors).")
        lines.append(f"- Mean full-label domain-informed R2 is {model.R2_mean:.3f}. " +
                     ("This is positive held-out explanatory performance for the recorded aggregate target."
                      if model.R2_mean > 0 else
                      "This does not establish useful squared-error reconstruction of the recorded target."))
    if "without_quantity" in full.index and "data_driven" in full.index:
        improvement = 1-full.loc["data_driven", "MAE_mean"]/full.loc["without_quantity", "MAE_mean"]
        lines.append(f"- At full labels, adding confirmed quantity changes unconstrained-NN MAE by "
                     f"{improvement:+.1%} improvement at the same aggregation level. This ablation shares "
                     "the quantity-model architecture and training settings; it was not optimized separately.")
    machine_table = pd.read_csv(out / "per_machine_metrics_aggregated.csv")
    machine_full = machine_table[machine_table.target.eq("elapsed_hours") &
                                 machine_table.completeness.eq(max(levels)) &
                                 machine_table.method.eq("domain_informed")]
    lines += ["", "Full-label domain-informed performance differs across presses:", "",
              "| Press | Test aggregates | Test orders | MAE h | R2 |",
              "|---|---:|---:|---:|---:|"]
    for _, row in machine_full.sort_values("machine").iterrows():
        lines.append(f"| {row.machine} | {int(row.n_records_min)} | {int(row.n_orders_min)} | "
                     f"{row.MAE_mean:.3f} | {row.R2_mean:.3f} |")
    negative = machine_full[machine_full.R2_mean.lt(0)]
    if not negative.empty:
        descriptions = [f"{row.machine} (R2 {row.R2_mean:.3f}, {int(row.n_records_min)} aggregates)"
                        for _, row in negative.iterrows()]
        lines += ["", "Negative machine-level R2 remains for " + "; ".join(descriptions) +
                  ". Therefore positive pooled performance must not be presented as successful "
                  "identification on every press. Very small per-machine test subsets make these "
                  "diagnostics unstable; no unfavorable aggregate is removed."]
    lines += ["", "The quantity-aware ridge baseline uses the same encoded inputs and only observed training "
              "targets. It tests whether a simple quantity relationship explains the case. Successful quantity-based "
              "reconstruction alone does not establish an advantage for neural constraints. Per-machine diagnostics "
              "are reported for every selected press, including unfavorable results; no machine is selected using "
              "its test prediction error.", "",
              "### Previously unevaluated machines: secondary diagnostic", "",
              "The following fixed subdivision separates machines included in the earlier booking-level "
              "test from machines not previously evaluated by that model. The new-machine subset was "
              "defined by evaluation history, not by favorable errors. It remains one company and the "
              "same calendar period; it is a secondary diagnostic rather than an independent industrial "
              "confirmation. Full-label results are shown here; all levels are exported.", "",
              "| Test-machine subset | Method | Aggregates | Orders | MAE h | R2 |",
              "|---|---|---:|---:|---:|---:|"]
    subset_full = subset_aggregate[subset_aggregate.target.eq("elapsed_hours") &
                                   subset_aggregate.completeness.eq(max(levels))]
    for _, row in subset_full.sort_values(["evaluation_subset", "method"]).iterrows():
        lines.append(f"| {row.evaluation_subset.replace('_', ' ')} | {LABELS[row.method]} | "
                     f"{int(row.n_records_min)} | {int(row.n_orders_min)} | {row.MAE_mean:.3f} | "
                     f"{row.R2_mean:.3f} |")
    if not convergence.empty:
        capped = int(convergence.epoch_budget_reached.sum())
        worsened = int((convergence.validation_normalized_mae_at_end >
                        1.1 * convergence.validation_normalized_mae_at_best).sum())
        lines += ["", "### Optimization and convergence", "",
                  f"Of {len(convergence)} evaluation neural fits, {capped} reached the fixed "
                  f"{int(settings['max_epochs'])}-epoch budget. Restored checkpoints range from epoch "
                  f"{int(convergence.best_epoch.min())} to {int(convergence.best_epoch.max())}. "
                  f"For {worsened} fits, final observed-validation error was more than 10% above the "
                  "selected checkpoint; predictions use the restored checkpoint. This diagnostic flags "
                  "late deterioration, not an independent test of overfitting. Finite loss histories "
                  "and repeatability do not establish full optimization convergence.", ""]
        tuning_path = out / "tuning_trials.csv"
        if tuning_path.exists():
            tuning = pd.read_csv(tuning_path)
            if "epochs_run" in tuning:
                count = int(tuning.epochs_run.ge(settings["max_epochs"]).sum())
                lines += [f"During complete-label calibration, {count}/{len(tuning)} candidate/weight "
                          "fits also reached the epoch cap. The bounded search and common budget are "
                          "kept fixed after test evaluation; conclusions are conditional on this "
                          "optimization budget, and larger-budget performance remains untested.", ""]
        lines += ["`convergence_diagnostics.csv` records checkpoint-specific train/validation errors "
                  "and losses. `learning_curves.png/pdf` shows both neural methods at full and lowest "
                  "label availability for preselected seed 101. Training/validation errors use only "
                  "observed targets. Constraint penalties are shown unweighted for both methods, "
                  "including the unconstrained model as a diagnostic.", ""]
    lines += ["", "## Observation protocol and limitations", "",
              "Duration labels are masked after aggregation; quantity, machine, product and operation descriptors "
              "remain available. This is missing duration information, not deletion of arbitrary raw events. "
              "Training and early-stopping validation labels are reduced, while complete test targets remain "
              "evaluation-only. Normalization uses observed training targets. Constraints use predictions rather "
              "than hidden observed times.", ""]
    counts = [c for c in ["completeness", "n_train", "n_test", "n_observed_train_per_target",
                          "n_observed_validation_per_target"] if c in metrics]
    if len(counts) > 1:
        count_table = metrics[metrics.target.eq("elapsed_hours") & metrics.method.eq("domain_informed") &
                              metrics.split.eq("test")][counts].drop_duplicates().sort_values("completeness")
        count_table.to_csv(out / "label_budget.csv", index=False)
        lines += ["Exact available-label counts are in `label_budget.csv`; percentages refer to duration "
                  "labels, not all operational information.", ""]
    calibration = settings.get("configuration_calibration", "full_validation")
    if calibration == "full_validation":
        lines += ["Architecture and constraint-weight selection use complete development validation before the "
                  "missing-label comparison. Thus sparse fits have sparse early stopping but benefit from "
                  "full-label configuration calibration. They do not establish the same performance when only "
                  "the stated label fraction is available throughout all model development.", ""]
    else:
        lines += [f"Configuration calibration protocol: `{calibration}`; see the frozen configuration "
                  "and tuning artifacts for the labels used in model selection.", ""]
    observed_violation = workstation_violations(reference[["elapsed_observed", "booked_observed"]].to_numpy(), tolerance)
    lines += [f"The nonnegativity and booked-within-window constraints are soft penalties. The fixed "
              f"{tolerance:.3f}-hour accounting allowance is approximate at aggregate level: constituent "
              "rounding errors can accumulate, and overlapping bookings do not prove physical single-server "
              f"capacity. The observed test accounting violation fraction is "
              f"{observed_violation['any_violation_fraction']:.1%}. No target is removed merely to make "
              "a proposed constraint hold. No individual-constraint ablation establishes which penalty causes "
              "an error difference.", "",
              "Seed means and standard deviations describe repeated masks and optimization on the **same "
              "industrial holdout**. They are not confidence intervals or independent industrial replications. "
              "Order-disjoint chronological splitting limits leakage, but explicit parent-child order fields "
              "are empty and broader commercial relationships remain unresolved. Some later-period Company 2 "
              "bookings were previously examined in the earlier booking-level study. Changing aggregation "
              "does not create a previously unseen external dataset; this case is exploratory industrial "
              "evidence, not independent confirmation or a significance claim across companies.", "",
              "The export is a live, nontransactional snapshot. Summed confirmations and times describe "
              "recorded accounting, not guaranteed complete physical production. Quantity corrections, ambiguous "
              "press assignments, invalid intervals and cohort exclusions are detailed in the cohort audit. "
              "A small number of order-operation aggregates limits uncertainty assessment and machine-level "
              "interpretation. All held-out points remain visible in the prediction figures.", "",
              "## Reproduction and artifacts", "",
              "```powershell", '$env:PYTHONPATH = "$PWD/src"',
              "python -m pytest -q --basetemp=.pytest_tmp",
              "python -m linn.experiments.company2_case --config configs/company2_case.json", "```", "",
              "`metrics.csv` contains both targets and observed-validation/test metrics; "
              "`metrics_aggregated.csv` contains seed means/SD; `paired_comparison.csv` and "
              "`effect_summary.csv` contain paired neural effects. `per_machine_metrics*.csv` report "
              "all selected machines on complete test references. `test_subset_metrics*.csv` give the "
              "predefined previous/new-machine subdivision. `paper_table.tex` exports the elapsed "
              "comparison. Figures are available as PNG and PDF: `case_overview`, `predicted_observed`, "
              "and `per_machine_comparison`. Scatter plots use median predictions across seeds only as "
              "an illustration; reported metrics average individual fits. `convergence_diagnostics.csv` "
              "and `learning_curves.png/pdf` document the frozen training budget and selected checkpoints.", "",
              "Industrial record-level artifacts contain company-derived information and should remain "
              "private until publication authorization and anonymization have been settled."]
    if cohort:
        lines += ["", "The machine-selection rationale, split sizes and exclusion counts are retained "
                  "in `cohort_summary.json` and the preprocessing audit; raw sources are unchanged."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
