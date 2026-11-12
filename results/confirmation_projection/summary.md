# Post-hoc feasibility comparisons on reproduced confirmation worlds

These are added descriptive analyses of the existing 20-world design. The original protocol document was missing; its execution specification was reconstructed before the present rerun. The previously reported primary outcome was already known. This is a reproduction and post-hoc extension, not new prospective confirmation.

Both training and validation labels are reduced: at 5%, each target has 18 training and six validation labels per workstation. Ridge uses exactly the archived training masks and training-fitted feature representation, fixed alpha 1 and an unpenalized intercept. Neural predictions and checkpoints are unchanged. Both output corrections yield zero violations; projection may alter elapsed as well as booked time.

## Macro metrics

| Labels | Method | Elapsed MAE (h) | Booked MAE (h) | Joint violations (%) |
|---:|---|---:|---:|---:|
| 100% | ridge | 0.243665 | 0.242294 | 3.556 |
| 100% | ridge_projection | 0.239096 | 0.235344 | 0.000 |
| 100% | ridge_clipping | 0.239096 | 0.235344 | 0.000 |
| 100% | data_driven | 0.089515 | 0.085779 | 0.097 |
| 100% | data_driven_projection | 0.089511 | 0.085779 | 0.000 |
| 100% | data_driven_clipping | 0.089515 | 0.085780 | 0.000 |
| 100% | domain_informed | 0.089582 | 0.085753 | 0.153 |
| 100% | domain_informed_projection | 0.089576 | 0.085752 | 0.000 |
| 100% | domain_informed_clipping | 0.089582 | 0.085752 | 0.000 |
| 5% | ridge | 0.317611 | 0.324417 | 43.125 |
| 5% | ridge_projection | 0.299165 | 0.296134 | 0.000 |
| 5% | ridge_clipping | 0.313731 | 0.299200 | 0.000 |
| 5% | data_driven | 0.366464 | 0.352360 | 44.069 |
| 5% | data_driven_projection | 0.330946 | 0.330939 | 0.000 |
| 5% | data_driven_clipping | 0.353639 | 0.342985 | 0.000 |
| 5% | domain_informed | 0.296090 | 0.288746 | 25.153 |
| 5% | domain_informed_projection | 0.289162 | 0.282157 | 0.000 |
| 5% | domain_informed_clipping | 0.293972 | 0.283111 | 0.000 |

## Descriptive paired elapsed-MAE reductions at 5% labels

Effects first average paired relative workstation reductions within a world, then average worlds. They need not equal ratios of macro MAEs. Intervals resample 20 complete worlds 20,000 times; no multiplicity correction is applied.

| Candidate | Baseline | Mean reduction (%) | Approximate 95% interval (%) | Positive worlds |
|---|---|---:|---|---:|
| domain_informed | data_driven | 18.80 | [13.88, 23.82] | 19/20 |
| domain_informed | data_driven_clipping | 16.86 | [11.99, 21.82] | 19/20 |
| domain_informed | data_driven_projection | 11.62 | [7.47, 16.09] | 19/20 |
| domain_informed | ridge | 6.97 | [-1.81, 15.26] | 14/20 |
| domain_informed | ridge_clipping | 5.99 | [-2.87, 14.31] | 13/20 |
| domain_informed | ridge_projection | 2.17 | [-6.80, 10.51] | 12/20 |
| domain_informed_projection | data_driven_clipping | 18.64 | [13.78, 23.56] | 19/20 |
| domain_informed_projection | data_driven_projection | 13.66 | [9.62, 17.98] | 19/20 |
| domain_informed_projection | ridge_clipping | 8.21 | [-0.24, 16.21] | 13/20 |
| domain_informed_projection | ridge_projection | 4.52 | [-4.04, 12.46] | 12/20 |

Ridge regularization was fixed rather than optimized. The generator family and workstation parameters are unchanged; these results do not establish industrial generalization. Comparing projected predictors on both sides is necessary when deployment requires guaranteed feasibility. Projection guarantees feasibility, not an improvement in every target-specific MAE.

Reproduce with `python tools/evaluate_confirmation_projection.py` after the confirmation runner finishes. All input SHA-256 hashes and additional intervals are retained in this directory. No original source artifacts are modified.
