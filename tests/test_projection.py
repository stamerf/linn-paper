import numpy as np
import pytest

from linn.evaluation.projection import project_durations, clip_durations


def test_projection_handles_accounting_boundary_and_corner():
    p = np.array([[2., 4.], [-10., 5.], [2., -3.], [-1., -.5], [2., 1.]])
    np.testing.assert_allclose(project_durations(p, .2),
                               [[2.9, 3.1], [0., .2], [2., 0.], [0., 0.], [2., 1.]])
    np.testing.assert_allclose(clip_durations(p, .2),
                               [[2., 2.2], [0., .2], [2., 0.], [0., 0.], [2., 1.]])


def test_projection_satisfies_nearest_point_optimality_and_idempotence():
    rng = np.random.default_rng(712)
    p = rng.normal(size=(500, 2)) * 4
    tolerance = .02
    q = project_durations(p, tolerance)
    assert (q >= 0).all()
    assert (q[:, 1] <= q[:, 0] + tolerance + 1e-12).all()
    np.testing.assert_allclose(project_durations(q, tolerance), q)
    # Projection's variational inequality against independently drawn feasible
    # points verifies the closest-point property, not just feasibility.
    feasible = np.maximum(rng.normal(size=(500, 2)) * 5, 0.)
    feasible[:, 1] = np.minimum(feasible[:, 1], feasible[:, 0] + tolerance)
    assert (np.sum((p-q) * (feasible-q), axis=1) <= 1e-10).all()


@pytest.mark.parametrize('p,tolerance', [([[1., np.nan]], 0), ([[1., 2.]], -.1),
                                       ([[1., 2.]], np.inf), ([1., 2.], 0)])
def test_projection_rejects_invalid_inputs(p, tolerance):
    with pytest.raises(ValueError):
        project_durations(p, tolerance)


def test_masked_ridge_matches_independent_objective_and_ignores_hidden_targets():
    from tools.evaluate_projection_baselines import ridge_predictions
    rng = np.random.default_rng(103)
    x = rng.normal(size=(20, 4))
    y = rng.normal(size=(20, 2)) + 7
    test_x = rng.normal(size=(5, 4))
    mask = rng.random((20, 2)) > .5
    actual = ridge_predictions(x, y, mask, test_x, alpha=1.7)
    altered = y.copy()
    altered[~mask] = np.nan
    np.testing.assert_array_equal(actual, ridge_predictions(x, altered, mask, test_x, alpha=1.7))
    for j in range(2):
        design = np.column_stack([np.ones(mask[:, j].sum()), x[mask[:, j]]])
        penalty = np.diag([0., 1.7, 1.7, 1.7, 1.7])
        coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y[mask[:, j], j])
        expected = np.column_stack([np.ones(5), test_x]) @ coefficient
        np.testing.assert_allclose(actual[:, j], expected, atol=1e-12)
