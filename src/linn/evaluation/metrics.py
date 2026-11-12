import numpy as np


def regression_metrics(observed, predicted):
    y, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require matching finite nonempty vectors")
    error = p-y
    denom = np.square(y-y.mean()).sum()
    return {"MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.square(error).mean())),
            "R2": float(1-np.square(error).sum()/denom) if denom > 0 else None,
            "WAPE": float(np.abs(error).sum()/np.abs(y).sum()) if np.abs(y).sum() > 0 else None,
            "mean_bias": float(error.mean()),
            "quantile_mae": float(np.abs(np.quantile(y,[.1,.5,.9])-np.quantile(p,[.1,.5,.9])).mean())}


def workstation_violations(prediction, tolerance=0.02):
    p = np.asarray(prediction, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or len(p) == 0:
        raise ValueError("Expected finite [N, 2] predictions")
    v = np.column_stack([np.maximum(-p[:,0],0), np.maximum(-p[:,1],0),
                         np.maximum(p[:,1]-p[:,0]-tolerance,0)])
    return {"any_violation_fraction": float((v > 1e-7).any(axis=1).mean()),
            "mean_violation_hours": v.mean(axis=0).tolist(),
            "violation_fractions": (v > 1e-7).mean(axis=0).tolist()}
