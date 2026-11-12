# Reconstructed synthetic confirmation execution specification

This document was reconstructed on 10 November 2026 from the checked-in
`configs/synthetic_confirmation.json`, `configs/workstation_synthetic.json`,
`results/synthetic/selected_configs.json`, experiment code, and existing manuscript.
The earlier protocol document and original confirmation artifacts were absent
from this checkout. This file documents the reproduction performed now; it is
not a recovered original protocol, an external preregistration, or evidence about
the chronology of the earlier experiment. The prior reported result was already
known before this reproduction. Model choices and seeds remain those in the
existing configuration, without tuning on the reproduced test scores.

## Design retained from the checked-in configuration

Generate the 20 worlds enumerated in `configs/synthetic_confirmation.json`, each
containing three workstations and 600 jobs per workstation. Keep chronological
partitions of 360 training, 120 validation, and 120 test jobs. The generator,
workstation parameters, input features, and selected architectures and optimizer
settings are inherited unchanged from the existing files. A world is the unit
of replication; its three workstations and jobs are not independent worlds.

Evaluate complete labels and 5% labels, with each target channel masked
independently in both training and validation. The sparse condition retains
18 training and six validation labels per target and workstation. Keep every
input record and all test targets. Fit encoders on training inputs and target
scales on observed training targets only. Checkpoint selection uses retained
validation labels only. Hidden target values never enter scaling, fitting,
or checkpoint selection.

Pair the data-driven and domain-informed predictors using the same inputs,
masks, initialization, batches, optimizer settings, and epoch budget. The
domain-informed model adds non-negativity and booked-within-elapsed penalties
with zero accounting tolerance. Use the inherited weights 0.1, 0.1, and 1.0 for
SIM_A, SIM_B, and SIM_C. Limit fitting to 160 epochs, patience 25, and minimum
normalized-validation-MAE improvement 0.0001. Restore the best qualifying
checkpoint. Run all 240 configured fits and the two existing repeat checks.

## Estimation and reproduction record

The inherited primary comparison is elapsed-duration MAE at 5% labels. Compute
relative improvement for each paired workstation, average workstations equally
within each world, and average the 20 world effects. Use 20,000 whole-world
bootstrap resamples with seed 61031 for the approximate percentile interval.
The inherited practical threshold is 5%. Complete-label, booked-target,
per-workstation, and consistency results remain secondary descriptions. The
reproduction does not establish new prospective confirmation because its
earlier reported outcome was already known.

Retain every configured world and disclose optimization-budget hits. Save
environment, protocol/source hashes, masks, preprocessing metadata, references,
histories, predictions, world effects, and interval summaries. Verify the hashes
of raw data and prior synthetic/industrial baseline artifacts remain unchanged.
The runner's protocol hash includes this reconstructed document and therefore
identifies the present reproduction rather than an original protocol snapshot.

## Added post-hoc projection comparison

The new comparator analysis is exploratory even though it uses the same 20
worlds and sparse-validation design. Reuse restored neural predictions without
retraining or selecting checkpoints on projected scores. Fit fixed-alpha Ridge
(alpha 1, unpenalized intercept) separately to each target using the identical
observed training masks and training-fitted feature representation. No test
labels or new tuning enter fitting. Compare raw predictions, Euclidean
projection in hours onto d>=0, b>=0, b<=d, and one-sided clipping that first
sets d=max(d,0), then b=min(max(b,0),d). Also report a projected domain-informed
predictor so applications requiring exact feasibility can compare feasible
outputs on both sides.

Report MAE for both channels and joint violations separately. Summarize paired
relative reductions by averaging workstation effects within each world, then
bootstrap whole worlds using the inherited seed and 20,000 resamples. All
additional intervals are descriptive and are not adjusted for the multiple
comparisons. They cannot be presented as newly preregistered primary results.
