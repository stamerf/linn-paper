# Synthetic workstation diagnostic

This is a controlled learnability check, not a reproduction of Company 2 and not MS3.

## Design

Three single-server machines process 600 synthetic batch jobs each. Each has 360 training, 120 validation and 120 test jobs, chronologically disjoint. Rates are 40, 65 and 90 units/hour. Product work, planned quantity, a small calendar effect and batch nonlinearity determine mean processing time. Independent 5% lognormal variation and positive pauses make the labels noisy. A FCFS arrival/start/completion recurrence supplies plausible serial event sequences. No physical capacity constraint is added to the neural model.

The neural architecture, loss, trainer and workstation constraints are reused. The only extra input in the full-feature models is planned quantity, standardized on training data. The limited-feature neural baseline uses the same product/operation/calendar feature types as the industrial run, without quantity. It is separately validation-tuned to avoid disadvantaging that baseline. The full-feature constrained and unconstrained models share architecture, optimizer settings, initial seed and minibatch ordering. Early-stopping epochs may differ. All configurations are frozen before evaluating the test jobs.

The generator's conditional mean is an audit-only reference, not a learned method or a literal lower bound on finite-sample MAE. The network never receives oracle values, hidden rates, realized pauses or queue states.

## Held-out elapsed-time results

| Machine | Method | MAE h (mean ± seed SD) | R2 | WAPE | Any violation |
|---|---|---:|---:|---:|---:|
| SIM_A | data_driven | 0.127 ± 0.004 | 0.985 | 4.5% | 0.0% |
| SIM_A | domain_informed | 0.127 ± 0.004 | 0.985 | 4.5% | 0.0% |
| SIM_A | limited_features | 1.097 ± 0.013 | 0.274 | 39.2% | 0.3% |
| SIM_A | median | 1.250 ± 0.000 | -0.078 | 44.6% | 0.0% |
| SIM_A | oracle_mean | 0.112 ± 0.000 | 0.989 | 4.0% | 0.0% |
| SIM_B | data_driven | 0.066 ± 0.003 | 0.994 | 3.9% | 0.6% |
| SIM_B | domain_informed | 0.066 ± 0.003 | 0.994 | 3.9% | 0.6% |
| SIM_B | limited_features | 0.753 ± 0.005 | 0.365 | 44.3% | 0.0% |
| SIM_B | median | 0.895 ± 0.000 | -0.024 | 52.6% | 0.0% |
| SIM_B | oracle_mean | 0.058 ± 0.000 | 0.995 | 3.4% | 0.0% |
| SIM_C | data_driven | 0.060 ± 0.003 | 0.987 | 5.0% | 0.3% |
| SIM_C | domain_informed | 0.059 ± 0.002 | 0.987 | 4.9% | 0.0% |
| SIM_C | limited_features | 0.514 ± 0.006 | 0.262 | 43.1% | 0.0% |
| SIM_C | median | 0.557 ± 0.000 | -0.006 | 46.6% | 0.0% |
| SIM_C | oracle_mean | 0.050 ± 0.000 | 0.990 | 4.2% | 0.0% |

## Interpretation

Learnability criterion fixed before evaluation: test R2 >= 0.8 and WAPE <= 15% for every evaluation seed. This is separate from a claim that constraints improve accuracy.
- SIM_A: domain-informed learnability criterion met. Providing quantity reduces unconstrained MAE by 88.4%; adding constraints to full features changes MAE by +0.0% improvement (negative means worse).
- SIM_B: domain-informed learnability criterion met. Providing quantity reduces unconstrained MAE by 91.2%; adding constraints to full features changes MAE by +0.7% improvement (negative means worse).
- SIM_C: domain-informed learnability criterion met. Providing quantity reduces unconstrained MAE by 88.4%; adding constraints to full features changes MAE by +1.6% improvement (negative means worse).

The full-feature versus limited-feature comparison tests the value of a known independent process driver. The constrained versus unconstrained comparison tests the added value of soft feasibility penalties. Good full-feature prediction alone does not establish that domain constraints caused it. If unconstrained predictions already satisfy the broad inequalities, those inequalities provide little extra information.

## Scientific limits

This simulated mapping was deliberately made identifiable from the full input set. Its clean job grain, observed planned quantity and stable product effects are stronger assumptions than the industrial booking export supports. Do not transfer the synthetic accuracy to Company 2 or reinterpret retrospective quantity confirmations as planned batch quantities. One simulation seed and three optimization seeds are used: the SD is not uncertainty across simulation worlds. No missingness experiment, transport, setup, inspection, failures or real-calendar shifts are modeled. Simulated queue waits are audit-only. The experiment does not validate learned queue dynamics or single-server feasibility of predicted completions.

## Validation and reproduction

The generator reproduces exactly. Both full-feature neural methods were trained twice at the first evaluation seed on each machine and required identical histories and parameters. Raw Company 2 inputs and all existing MS2 result bytes were checked unchanged. See validation.json, source_hashes.json, selected_configs.json and tuning_trials.csv.

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m pytest -q --basetemp=.pytest_tmp
python -m linn.experiments.workstation_synthetic --config configs/workstation_synthetic.json
```

Outputs in results/synthetic/: simulated_bookings.csv, simulation_summary.csv, metrics.csv, metrics_aggregated.csv, paired_comparison.csv, predictions.csv, training_history.csv, training_diagnostics.csv, models/, encoder JSONs, figures/ and paper_table.tex. Use synthetic_comparison and the per-machine reconstruction panels together. Industrial MS3 remains unstarted.
