# Publication interpretation of the completed synthetic experiment

The experiment supports the bounded claim that soft accounting constraints improve
identification under severe training-label sparsity in this synthetic generator.
It does not establish improved industrial identification or robustness to arbitrary
missing information.

At 5% label availability, macro elapsed MAE decreases from 0.330 to 0.222 hours
(19.8 to 13.3 minutes). The mean paired improvement is 31.3%; the reduction computed
from aggregate MAEs is 32.9%. These are different aggregations, not contradictory
estimates. Mean paired booked-time improvement is 29.5%. Elapsed improvement is
32.7%, 25.6%, and 35.7% for SIM_A, SIM_B, and SIM_C respectively. Each machine has
positive effects in all three worlds after averaging its two paired repeats.
The fraction of predictions violating at least one accounting constraint decreases
from 44.9% to 19.4%; soft constraints therefore improve but do not ensure feasibility.

At 10% labels, mean paired elapsed improvement is 7.8%, but SIM_C has a negative
world-level effect in one world. At 25% it is only 1.4%, and at 50% it is 0.2%.
At 75% and 100%, constrained errors are marginally worse. The evidence supports
a benefit at severe sparsity, not general superiority at every completeness level.

Quantity removal gives essentially equal errors for the methods: about 0.671 hours
at 25% quantity availability and 0.797 hours at 5%, versus about 0.085 hours with
complete inputs. Nonnegativity and booked-versus-elapsed consistency cannot recover
unobserved workload. This negative control is useful for the paper because it
distinguishes missing supervision from missing predictive information.

## Qualification and optimization diagnostics

All 120 validation jobs retain both targets. At 5% training-label availability,
there are 18 training labels per target plus 120 validation labels per target.
Describe the regime as sparse training supervision with complete validation, never
as 5% of all observations. Only three stochastic worlds from one generator family
were evaluated; reported standard deviations are descriptive, not confidence intervals.

At 5%, seven of 18 constrained fits and one of 18 unconstrained fits reach the
160-epoch budget. The selected epoch averages 116 versus 69 respectively. Thus
convergence is not established for every fit and conclusions are specific to the
declared budget. No test-driven retuning was performed. A future preregistered
budget-sensitivity experiment should use new held-out worlds. No individual
constraint ablation was run, so the contribution of each penalty is unresolved.

## Recommended paper artifacts and next experiments

Use `figures/degradation_overview.pdf` for the main result, `per_machine_labels.pdf`
for machine consistency, and `paper_table.tex` for numerical results. Shading/error
bars represent standard deviation across three worlds after averaging paired
repeats. Reconstruction examples are illustrations from the prespecified first
world/machine/seed, not evidence on their own. CSV files preserve both targets,
negative effects, paired results, and convergence diagnostics.

Before a broad publication claim, evaluate reduced validation-label availability,
individual constraint ablations, and additional generator families or parameter
regimes. Treat these as new experiments with frozen protocols and fresh evaluation
worlds. Industrial validation remains a separate unresolved requirement; industrial
MS3 has not been started by this synthetic study.
