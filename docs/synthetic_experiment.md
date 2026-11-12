# Synthetic workstation diagnostic

This separate experiment tests whether the existing model can learn a process whose
drivers are observable. It does not reproduce Company 2 or start industrial MS3.

Three continuously available single-server machines process 600 batch jobs each.
Rates are 40, 65 and 90 units/hour. Arrivals are exponential with mean 3.5 hours;
start is the maximum of arrival and the previous completion. Each job has a uniformly
sampled quantity of 20–160 and one of three products with work factors 0.7, 1 and 1.5.

The conditional mean active duration is:

`quantity / rate * product_factor * (1 + 0.08 sin(2π start_hour/24)) * (1 + 0.08 (quantity/100)^2)`.

Multiply this by mean-one lognormal noise with log standard deviation 0.05.
Elapsed time adds a pause of `0.04 + 0.0004 quantity + Exponential(mean=0.025)` hours.
The event generator guarantees nonnegative durations and serial capacity. Neither
its rate equation nor its coefficients are supplied to the neural model or constraints.
Queue waits and conditional-mean oracle columns are for audit only.

Each machine uses 360 training, 120 validation and 120 test jobs in chronological order.
Encoding and scaling fit training data only. Two network configurations and two positive
constraint weights are validation-tuned at two seeds; three new optimization seeds
are evaluated after choices are frozen. The existing model, trainer and constraints
are reused. Full features add independently known planned quantity to product,
operation and start-calendar inputs. A separately tuned limited-feature network omits
quantity. Median predictions and the known generator mean provide references.

The constrained and unconstrained full-feature networks share architecture, initial
seed and minibatch ordering. Their stopping epochs can differ. Penalties enforce soft
nonnegativity and booked time no greater than elapsed time, with zero rounding tolerance.
They do not enforce serial feasibility of predicted completions.

The configured learnability criterion is test R2 ≥ 0.8 and WAPE ≤ 15% for every seed.
WAPE is total absolute error divided by total observed time, not classification accuracy.
Good prediction with quantity alone does not demonstrate a benefit from constraints.
One simulation seed does not establish robustness across different simulated worlds.
The clean batch grain and known quantity are stronger assumptions than Company 2 supports;
retrospective quantity confirmations must not be treated as planned batch quantities.

Run from the repository root:

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m pytest -q --basetemp=.pytest_tmp
python -m linn.experiments.workstation_synthetic --config configs/workstation_synthetic.json
```

Results and interpretations are in `results/synthetic/summary.md`, with CSV metrics,
predictions, source hashes, checkpoints, and PNG/PDF figures in that directory.
