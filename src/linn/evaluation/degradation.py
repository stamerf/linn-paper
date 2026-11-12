"""Nested, fixed-count observation masks independent of values and row ordering."""
import numpy as np


def observation_mask(record_ids, completeness, seed, n_columns=1, whole_record=False):
    ids = np.asarray(record_ids, dtype=str)
    if ids.ndim != 1 or len(set(ids)) != len(ids):
        raise ValueError("Record IDs must be unique")
    if not 0 <= completeness <= 1 or n_columns < 1:
        raise ValueError("Invalid completeness or number of columns")
    count = int(np.floor(len(ids)*completeness))
    ordered = np.argsort(ids, kind="stable")
    rng = np.random.default_rng(seed)
    mask = np.zeros((len(ids), n_columns), dtype=bool)
    for col in range(1 if whole_record else n_columns):
        selected = ordered[rng.permutation(len(ids))[:count]]
        mask[selected,col] = True
    if whole_record:
        mask[:] = mask[:,[0]]
    return mask


class ObservedMedianImputer:
    """Fit only observed training inputs; retain an explicit missingness indicator."""
    def fit(self, values, observed):
        values = np.asarray(values,dtype=float);observed=np.asarray(observed,dtype=bool)
        if values.shape != observed.shape or not observed.any() or not np.isfinite(values[observed]).all():
            raise ValueError("Need finite observed training values")
        self.median = float(np.median(values[observed]))
        return self

    def transform(self, values, observed):
        values=np.asarray(values,dtype=float);observed=np.asarray(observed,dtype=bool)
        if values.shape != observed.shape or not np.isfinite(values[observed]).all():
            raise ValueError("Invalid observed input values")
        return np.where(observed,values,self.median), (~observed).astype(float)
