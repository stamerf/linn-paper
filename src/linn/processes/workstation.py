import numpy as np
from .base_process import ProcessSpec
from linn.constraints.workstation_constraints import WorkstationConstraints
from linn.data.workstation_dataset import TARGETS


def workstation_spec(train, tolerance=0.02):
    if train.empty or not train["split"].eq("train").all():
        raise ValueError("Scale fitting requires training data only")
    scales = train[list(TARGETS)].median().to_numpy(dtype=float)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Targets need positive finite training medians")
    return ProcessSpec("workstation_booking", TARGETS, WorkstationConstraints(),
                       {"target_scale": scales.tolist(), "constraint_scale": float(scales[0]),
                        "rounding_tolerance_hours": tolerance})
