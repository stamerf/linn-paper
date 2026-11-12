# Post-hoc feasibility baselines

This exploratory comparison reuses saved synthetic degradation predictions. It is separate from the 20-world confirmation; confirmation and industrial prediction artifacts are unavailable in this checkout. No original artifacts are changed.

Three worlds, three machines and two paired mask/optimizer seeds give 18 paired runs at each training-label level. All neural fits used 120 complete validation labels per target and machine. The table averages runs equally; world SD first averages machines and seeds and is descriptive, not a confidence interval.

Ridge uses the same train-only encoded features and exactly the archived target-channel masks, an unpenalized intercept and fixed alpha=1. It is fitted by float64 SVD; hidden labels are removed before fitting. No test labels enter features, fitting or parameter selection. Alpha=1 is a fixed comparator choice, not a validation-tuned optimum. Neural checkpoints remain those of the original study.

Projection minimizes squared distance between duration pairs in original hours over d>=0, b>=0, b<=d (epsilon=0). It adjusts both channels when needed. The supplementary clipping sensitivity sets d=max(d,0) then b=min(max(b,0),d), preserving nonnegative elapsed predictions. All projected and clipped predictions have zero joint violations. Projection guarantees feasibility, not lower per-target MAE.

| Labels | Method | Elapsed MAE (h) | Booked MAE (h) | Joint violation (%) |
|---:|---|---:|---:|---:|
| 100% | ridge | 0.254213 | 0.251812 | 5.278 |
| 100% | ridge_projection | 0.244586 | 0.238681 | 0.000 |
| 100% | data_driven | 0.085310 | 0.081291 | 0.000 |
| 100% | data_driven_projection | 0.085310 | 0.081291 | 0.000 |
| 100% | domain_informed | 0.085623 | 0.081883 | 0.046 |
| 100% | ridge_clipping | 0.244586 | 0.238681 | 0.000 |
| 100% | data_driven_clipping | 0.085310 | 0.081291 | 0.000 |
| 10% | ridge | 0.276524 | 0.280139 | 40.093 |
| 10% | ridge_projection | 0.264762 | 0.260395 | 0.000 |
| 10% | data_driven | 0.146454 | 0.145608 | 24.954 |
| 10% | data_driven_projection | 0.142269 | 0.140742 | 0.000 |
| 10% | domain_informed | 0.136115 | 0.134867 | 14.676 |
| 10% | ridge_clipping | 0.269854 | 0.258373 | 0.000 |
| 10% | data_driven_clipping | 0.146454 | 0.139974 | 0.000 |
| 5% | ridge | 0.318390 | 0.310168 | 42.685 |
| 5% | ridge_projection | 0.297770 | 0.289497 | 0.000 |
| 5% | data_driven | 0.330297 | 0.318233 | 44.907 |
| 5% | data_driven_projection | 0.293372 | 0.294837 | 0.000 |
| 5% | domain_informed | 0.221688 | 0.219306 | 19.352 |
| 5% | ridge_clipping | 0.313432 | 0.295209 | 0.000 |
| 5% | data_driven_clipping | 0.318691 | 0.307965 | 0.000 |

Paired relative reductions average per-run reductions, rather than ratios of aggregate MAEs.

| Labels | Target | Domain-informed reduction vs NN + Projection | vs Ridge + Projection |
|---:|---|---:|---:|
| 5% | booked_hours | 24.24% | 25.90% |
| 5% | elapsed_hours | 23.01% | 27.42% |
| 10% | booked_hours | 4.79% | 48.51% |
| 10% | elapsed_hours | 5.11% | 48.13% |
| 100% | booked_hours | -0.91% | 64.80% |
| 100% | elapsed_hours | -0.39% | 63.73% |

These descriptive comparisons use only three independently simulated worlds. They do not establish a general advantage over projection or industrial generalization. A soft penalty still leaves violations; applications requiring guaranteed feasibility must apply a correction or hard constraint regardless of predictive model.

Reproduce with `python tools/evaluate_projection_baselines.py`. The input manifest records SHA-256 hashes and the script verifies inputs remain unchanged.
