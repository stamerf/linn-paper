# Confirmatory synthetic results

Primary mean paired elapsed-MAE reduction: **18.8%**, approximate two-sided 95% world-bootstrap interval **[13.9%, 23.8%]**. The interval supports a positive mean effect. Its lower bound also exceeds the prespecified 5% practical threshold.

The primary outcome averages paired relative reductions across three machines within each world, then across 20 independent worlds. It is not the ratio of pooled MAEs. Bootstrap resampling preserves each whole world; jobs and machines are not treated as independent replications.

## Macro metrics

| completeness | target | method | MAE | RMSE | R2 | constraint_violation | budget_hits |
|---|---|---|---|---|---|---|---|
| 0.0500 | booked_hours | data_driven | 0.3524 | 0.4802 | 0.8051 | 0.4407 | 1 |
| 0.0500 | booked_hours | domain_informed | 0.2887 | 0.4055 | 0.8559 | 0.2515 | 4 |
| 0.0500 | elapsed_hours | data_driven | 0.3665 | 0.5027 | 0.7838 | 0.4407 | 1 |
| 0.0500 | elapsed_hours | domain_informed | 0.2961 | 0.4099 | 0.8538 | 0.2515 | 4 |
| 1.0000 | booked_hours | data_driven | 0.0858 | 0.1211 | 0.9880 | 0.0010 | 1 |
| 1.0000 | booked_hours | domain_informed | 0.0858 | 0.1211 | 0.9880 | 0.0015 | 1 |
| 1.0000 | elapsed_hours | data_driven | 0.0895 | 0.1243 | 0.9875 | 0.0010 | 1 |
| 1.0000 | elapsed_hours | domain_informed | 0.0896 | 0.1245 | 0.9875 | 0.0015 | 1 |

## Scope and interpretation

19 of 20 worlds have positive macro paired effects; the median world effect is 17.4%. World effects range from -0.4% to 39.2%.

At 5%, only 18 training and 6 validation labels per target are available per machine. Both methods use identical masks, inputs, initialization and optimization budgets. Test labels remain complete. Frozen choices originate from an earlier development world; none were selected on these test results. All 20 worlds are retained, with no optional stopping.

Only the elapsed-time effect at 5% is confirmatory. Complete-data control, booked-time, per-machine intervals and other metrics are descriptive secondary analyses. Percentile bootstrap coverage is approximate with 20 worlds. This local frozen protocol is not an externally registered study.

The generator family and machine parameters remain fixed. Evidence here does not establish industrial generalization, robustness to other generators, or each constraint's contribution. Early stopping with six labels per target is noisy; all budget hits are exported. Soft constraints do not guarantee feasibility. No industrial MS3 was run.

## Validation and artifacts

240 fits completed; 2 exact-repeat checks. All metrics are finite. Protocol, source hashes, environment, references, masks, scaling, histories, predictions, paired effects, confidence intervals, PNG/PDF figure and LaTeX table are saved here.

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m pytest -q --basetemp=.pytest_tmp
python -m linn.experiments.synthetic_confirmation --config configs/synthetic_confirmation.json
```
