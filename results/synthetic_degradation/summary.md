# Synthetic observation degradation results

This experiment tests the previously proposed soft accounting constraints on fresh synthetic worlds. It does not alter the industrial data or run industrial MS3.

## Frozen protocol

Three fresh simulation seeds (811, 977, 1231), three machines and two paired mask/optimizer seeds (101, 202) per world. Each machine has 360 training, 120 validation and 120 held-out test jobs. Architectures and positive constraint weights were frozen from the earlier validation-selected synthetic run (world 731), never tuned on these test samples. Both methods share inputs, masks, initialization, batch order and a 160-epoch / 25-patience budget. Stopping epochs may differ. There are 288 fits plus six exact-repeat checks.

**Label removal (main experiment):** independently mask elapsed and booked training labels at 100%, 75%, 50%, 25%, 10%, 5%. Exactly floor(n × fraction) observations per target are retained. Masks are nested across levels and independent of target values. All 360 feature rows remain available; constraints may operate on unlabeled rows. This is label sparsity, not wholesale row deletion. Train target scales use retained labels only. Hidden values are replaced by NaN before scale fitting and by masked placeholders in tensors. Constraints receive predictions and no target data.

**Complete validation anchor:** all 120 validation jobs retain both labels for early stopping. At 5%, the learner still receives 18 labels per target in training plus 120 per target in validation. Therefore this experiment measures sparse training supervision with complete calibration, not 5% of all development observations. This anchor is shared by both methods and can substantially moderate degradation.

**Quantity removal (separate experiment):** retain quantity for 100%, 25%, 5% of jobs in training, validation and test; duration labels stay complete. Replace missing quantities with the median of observed training quantities and include a missingness flag. Inputs at test time are degraded but evaluation labels remain complete. This changes information available at deployment and must not be conflated with training-label removal. Both mechanisms use the same complete control fit.

## Macro elapsed-time results

MAE is in hours. Means weight each machine and simulation world equally. SD is across three worlds after averaging the two mask/optimizer repeats within each world. It is descriptive, not a confidence interval.

| Mechanism | Available | Data-driven MAE | Domain-informed MAE | Paired improvement |
|---|---:|---:|---:|---:|
| labels | 100% | 0.085 ± 0.005 | 0.086 ± 0.005 | -0.4% |
| labels | 75% | 0.088 ± 0.005 | 0.088 ± 0.005 | -0.3% |
| labels | 50% | 0.090 ± 0.005 | 0.090 ± 0.005 | +0.2% |
| labels | 25% | 0.100 ± 0.003 | 0.099 ± 0.004 | +1.4% |
| labels | 10% | 0.146 ± 0.015 | 0.136 ± 0.009 | +7.8% |
| labels | 5% | 0.330 ± 0.052 | 0.222 ± 0.007 | +31.3% |
| quantity | 100% | 0.085 ± 0.005 | 0.086 ± 0.005 | -0.4% |
| quantity | 25% | 0.671 ± 0.018 | 0.671 ± 0.018 | -0.1% |
| quantity | 5% | 0.797 ± 0.008 | 0.797 ± 0.007 | -0.1% |

## Interpretation

Positive improvement means lower constrained MAE; negative means worse. A descriptive material-effect threshold of 5% was configured before the run. It is not a statistical significance test.
- labels: 4 of 15 machine/level combinations have at least 5% mean paired improvement and positive mean effects in all three worlds.
  - SIM_A: best +32.7% at 5%; worst -0.7% at 75%.
  - SIM_B: best +25.6% at 5%; worst -0.2% at 75%.
  - SIM_C: best +35.7% at 5%; worst -0.2% at 75%.
- quantity: 0 of 6 machine/level combinations have at least 5% mean paired improvement and positive mean effects in all three worlds.
  - SIM_A: best +0.0% at 25%; worst -0.0% at 5%.
  - SIM_B: best +0.0% at 25%; worst +0.0% at 5%.
  - SIM_C: best -0.2% at 5%; worst -0.3% at 25%.

The constraints do not encode quantity/rate equations. They only enforce nonnegativity and booked time within elapsed time. They cannot supply an unobserved workload. Constraint violations, per-target errors, per-world paired effects and negative effects are all retained in the exported tables. No constraint ablation was performed, so individual causal contributions cannot be claimed.

Limitations: one simple stationary generator family; only three independent simulated worlds; complete validation anchor for label removal; frozen hyperparameters rather than per-level optimization; fixed optimization budgets; simultaneous input loss at deployment in the quantity condition. No test-driven retuning was performed. Seed variation is not evidence of statistical significance.

## Validation and reproducibility

6 low-completeness fits were retrained exactly with matching parameters and histories. Every test prediction and reported metric is finite. Masks and fitted preprocessing are stored per condition. Run JSON files are written last, allowing interrupted runs to resume only with identical core code/configuration. Prior industrial and synthetic outputs are hash-protected.

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m pytest -q --basetemp=.pytest_tmp
python -m linn.experiments.synthetic_degradation --config configs/synthetic_degradation.json
```

Outputs: metrics.csv, metrics_aggregated.csv, macro_metrics.csv, paired_comparison.csv, paired_world_effects.csv, effect_summary.csv, protocol.json, environment.json, conditions/ (masks and scales), references/ (synthetic worlds), runs/ (predictions and loss histories), figures/ (PNG/PDF), and paper_table.tex. Use degradation_overview, per_machine_labels and reconstruction_examples for the paper; report the validation anchor explicitly. Industrial MS3 remains unstarted.
