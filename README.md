# LINN: domain-informed duration identification

Research code and synthetic experiment artifacts for jointly estimating elapsed
processing-related duration and booked processing time under incomplete labels.
The implementation separates simulation, preprocessing, neural prediction, masked
supervision, domain penalties, and evaluation. Post-hoc baselines compare ridge
regression and neural predictions with feasibility projection or clipping.

This public snapshot includes generated synthetic records, observation masks,
predictions, training histories, configurations, and aggregate results. Private
industrial records and results, local notebooks, and the unpublished manuscript
are excluded. Industrial adapters and tests remain as code; reproducing the real
industrial experiments requires separately authorized source data.

## Installation and tests

Use Python 3.10–3.12 in a virtual environment. From the repository root:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e '.[test]'
python -m pytest -q
```

`requirements-ms1.txt` records version pins used for the experiments; the archived
environment files distinguish the earlier and reproduced runs. Tests use generated
fixtures and included synthetic artifacts. Real industrial integration and
completed-industrial-artifact checks skip when their private inputs are absent.
These skips do not validate the industrial experiments.

## Included experiments and data

| Location | Contents |
|---|---|
| `configs/workstation_synthetic.json` | Generator and development-world configuration, seed 731. |
| `results/synthetic/` | Development records, predictions, metrics, and `selected_configs.json` used by later studies. |
| `results/synthetic_degradation/` | Three exploratory worlds, seeds 811, 977, and 1231; reduced training labels or quantity inputs, with complete validation labels. |
| `results/synthetic_confirmation/` | Reproduced 20-world study with complete labels or 5% labels in both training and validation. |
| `results/projection_baselines/` | Post-hoc comparisons on the three exploratory worlds. |
| `results/confirmation_projection/` | Post-hoc ridge, projection, and clipping comparisons on the 20 reproduced worlds. |

Within the degradation and confirmation folders, `references/` contains generated
world records, `conditions/` contains observation masks and fitted preprocessing
metadata, and `runs/` contains per-fit metrics, predictions, and training histories.
Top-level summaries and tables can be inspected without retraining. Identifiers
such as `SIM_A:00480` refer to simulated jobs, not industrial records.

The configurations enumerate all seeds, label budgets, and training settings.
`results/synthetic/selected_configs.json` preserves the development-selected neural
settings and is a required input to both later studies. Each synthetic machine
has 360 training, 120 validation, and 120 test jobs. At 5% training and validation
labels, each target retains 18 training and six validation observations.

## Recompute results without neural training

The included predictions and masks are sufficient for both projection analyses:

```bash
python tools/evaluate_projection_baselines.py
python tools/evaluate_confirmation_projection.py
```

These commands fit fixed-alpha ridge comparators and recompute metrics from saved
neural predictions; they do not retrain networks or reselect checkpoints. Exact
projection and one-sided clipping use predictions alone. Synthetic feasibility
requires nonnegative durations and booked time no greater than elapsed time.

Original neural summaries can also be regenerated:

```bash
python -m linn.experiments.synthetic_degradation --summarize-only
python -m linn.experiments.synthetic_confirmation --summarize-only
```

## New training runs

Keep archived results separate from new fits. For a new development-world run,
copy `configs/workstation_synthetic.json` to another file inside `configs/`, set
`output_dir` to a fresh immediate subdirectory of `results/`, then run:

```bash
python -m linn.experiments.workstation_synthetic --config configs/my_synthetic_run.json
```

The degradation and confirmation runners currently require their dedicated
output directories. For new full runs, use a separate working copy with those
archived result directories moved aside, retaining the development-selected
`results/synthetic/selected_configs.json`, then run:

```bash
python -m linn.experiments.synthetic_degradation --config configs/synthetic_degradation.json
python -m linn.experiments.synthetic_confirmation --config configs/synthetic_confirmation.json
```

Cached execution checks protocol and source hashes before resuming; changed
source bytes, path conventions, or experimental settings can invalidate a cache.
Private-data protection manifests from the original workspace are omitted from
this public snapshot. Fresh run directories let the runners record the current
public inputs without mixing new fits with the archived outcomes.

## Interpretation and provenance

Read the [generator description](docs/synthetic_experiment.md),
[exploratory protocol](docs/synthetic_degradation_protocol.md), and
[reconstructed execution specification](docs/synthetic_confirmation_protocol.md)
for information budgets and limitations. The 20-world execution reproduced an
earlier comparison whose outcome was already known. Its execution document was
reconstructed from existing code and configurations, not recovered as an original
preregistration. Added projection comparisons are post hoc and their intervals
are descriptive whole-world bootstrap intervals.

Evidence is conditional on the specified generator family. Random target removal
does not establish robustness to selective missingness or missing explanatory
inputs. Soft penalties may reduce violations; projection guarantees feasible
outputs. Synthetic gains do not establish industrial superiority or downstream
planning benefits.

Zenodo archival DOI: pending.
