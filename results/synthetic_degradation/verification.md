# Completed verification

Verified on 2026-10-03:

- Full test suite: 45 passed. Dependency deprecation warnings only.
- Saved-artifact test recomputes metrics from every fit's exported predictions;
  tolerance is 1e-6 for decimal serialization of float32 outputs.
- All 288 main fits complete; six low-completeness fits have exact repeated
  parameters and training histories.
- All three generated PNG figures visually inspected for legibility and layout.
- Full runner successfully resumed all saved fits with matching protocol hashes
  and regenerated reports without further training.
- Raw data and previous experiment outputs pass the runner's hash protection.
- `git diff --check` passes (Windows line-ending advisory only).

The runner's machine-readable protection and scope checks are in `validation.json`.
