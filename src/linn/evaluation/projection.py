"""Deterministic feasibility corrections in original duration units (hours)."""
import numpy as np


def _validate(prediction, tolerance):
    p = np.asarray(prediction, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('Expected finite [N, 2] duration predictions')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    return p


def project_durations(prediction, tolerance=0.):
    """Euclidean projection onto d >= 0, b >= 0, b <= d + tolerance.

    Both channels are measured in hours and receive equal distance weights.
    Unlike clipping only b, this nearest-point projection may change d as well.
    """
    p = _validate(prediction, tolerance)
    result = np.maximum(p, 0.)
    coupled = result[:, 1] > result[:, 0] + tolerance
    elapsed = np.maximum((p[coupled, 0] + p[coupled, 1] - tolerance) / 2., 0.)
    result[coupled, 0] = elapsed
    result[coupled, 1] = elapsed + tolerance
    return result


def clip_durations(prediction, tolerance=0.):
    """Clip negatives, then cap booked time while preserving corrected elapsed."""
    p = _validate(prediction, tolerance)
    result = np.maximum(p, 0.)
    result[:, 1] = np.minimum(result[:, 1], result[:, 0] + tolerance)
    return result
