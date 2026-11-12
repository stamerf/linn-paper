# Synthetic observation degradation protocol

This follow-up tests the robustness of the existing domain-informed learner. It
uses fresh realizations of the previously defined synthetic process, not Company 2.
The simulator and model physics are not strengthened to obtain a positive result.

## Fixed design

Configuration: `configs/synthetic_degradation.json`. Simulation seeds 811, 977 and
1231 differ from the original development world 731. Each has three machines and
600 jobs per machine, split 360/120/120 chronologically. These independent random
realizations share generator parameters; they do not test robustness across different
physical process families or unknown rate parameters.

The original synthetic validation-selected architectures and constraint weights are
frozen. Two optimizer/mask seeds (101 and 202) are paired between methods. Each fit
allows at most 160 epochs and stops after 25 epochs without validation improvement.
Initial parameters, feature tensors, minibatch order and observation masks are equal
within each method pair. Early-stopping epochs may differ. No new hyperparameter
search is performed at any degradation level or after viewing test performance.

All methods use a numeric quantity-missingness flag, including the complete-data
control, where it is always zero. This adds one input to the earlier experiment;
the complete control is retrained, not copied from the earlier result table.

## Two distinct mechanisms

**Training-label removal:** independently retain 100%, 75%, 50%, 25%, 10%, or 5%
of elapsed-duration and booked-duration labels. Each target retains exactly
`floor(n_train * completeness)` labels. The same deterministic permutations are
used across levels, giving nested masks. All training jobs and their non-target
features remain present, even when both duration labels are missing. This tests
sparse supervision, not merely reduced training-row count or loss of whole jobs.

**Quantity-input removal:** retain quantities for 100%, 25%, or 5% of jobs,
independently in training, validation, and test. Keep all duration labels. Impute
missing quantities with the median of the observed training quantities only, and
provide the missingness flag. This measures reduced information during both learning
and deployment. Held-out targets remain complete. It is a separate question from
missing supervision and must be plotted/interpreted separately.

The 100% conditions are identical and share one fitted control. There are 144
distinct conditions and 288 main fits, plus six exact-repeat fits at 5% labels.

## Complete calibration anchor

Training-label removal leaves all validation labels and validation inputs available.
At 5%, training has 18 labels per target, while validation has 120 labels per target.
Consequently, "5%" is **not** 5% of all development observations. Early stopping
still has substantial complete calibration data. This is a deliberate limitation
of this manageable first sweep and must appear in any paper description.

Test labels are never used for early stopping or model selection. Their observations
are the same for each completeness level within a simulation world and machine.
In the quantity condition, test inputs are masked but test labels are unchanged.

## Leakage safeguards

Target scales are recomputed using only retained training labels. Hidden values are
replaced by NaN before scaling, then masked placeholders in datasets. Metrics used
for training diagnostics respect the label mask; missing placeholders are not treated
as true zero durations. This corrects the earlier tuning helper, whose diagnostics
assumed complete labels. Complete-label behavior remains covered by the test suite.

Constraint inputs are empty: penalties use predicted nonnegativity and predicted
booked-versus-elapsed ordering only. They do not see observed targets, endpoints,
oracle generator outputs, rate parameters, or realized noise. The preprocessing
function drops end/oracle/truth columns. Quantity imputation uses only observed
training quantities. Tests replace hidden values with extreme numbers and require
unchanged tensors, scales, and preprocessing metadata.

Start timestamps and product identities remain observed in the label-removal
condition. This is a label-level observation mechanism; it does not claim to model
every industrial mechanism of missing start/completion events.

## Analysis

Save both targets' MAE, RMSE, R2, WAPE, quantile discrepancy, bias, and constraint
violations. Relative improvement is `(MAE_data_driven - MAE_domain_informed) /
MAE_data_driven`, calculated on each paired run before aggregation.

Average paired mask/optimizer repeats within each simulation realization. Then show means and
sample standard deviations across the three realizations. Do not treat six runs
as six independent simulation worlds. Macro results equally weight machines.
The preconfigured 5% mean improvement threshold is descriptive; consistency of
direction across three worlds is reported, not presented as a significance test.

Report positive and negative effects, and separate predictive accuracy from
constraint feasibility. The penalties are broad inequalities and cannot recover
an unknown workload. No ablation of individual penalties is included.

## Reproduction and interruption recovery

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m pytest -q --basetemp=.pytest_tmp
python -m linn.experiments.synthetic_degradation --config configs/synthetic_degradation.json
```

The runner writes each successful run JSON last, after predictions and history.
Repeating the command resumes completed fits only when the protocol and core code
match. Masks, scales, predictions, loss histories, selected epochs and model-parameter
hashes are retained. Model checkpoints are not required for regenerating the saved
figures; retraining is deterministic on the recorded CPU environment.

Reports can be regenerated without retraining:

```powershell
python -m linn.experiments.synthetic_degradation --summarize-only
```

All outputs use `results/synthetic_degradation/`. Prior raw data, MS2 and the original
synthetic artifacts are hash-protected. This is the authorized synthetic robustness
follow-up; industrial MS3 remains separate and unstarted.
