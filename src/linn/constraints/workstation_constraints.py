import torch
from .base_constraint import ConstraintSet


class WorkstationConstraints(ConstraintSet):
    """Soft booking accounting, not a single-server queue or cycle-time model.

    Inputs are predictions in hours. No targets or observed endpoints are needed.
    All penalties are dimensionless, using a common training-only hour scale.
    """
    def compute(self, prediction, batch, context):
        if prediction.ndim != 2 or prediction.shape[1] != 2:
            raise ValueError("Expected elapsed and booked predictions [N, 2]")
        scale = float(context["constraint_scale"])
        tolerance = float(context["rounding_tolerance_hours"])
        if scale <= 0 or tolerance < 0:
            raise ValueError("Invalid constraint scale or tolerance")
        elapsed, booked = prediction.unbind(dim=1)
        return {
            "elapsed_nonnegative": (torch.relu(-elapsed)/scale).square().mean(),
            "booked_nonnegative": (torch.relu(-booked)/scale).square().mean(),
            "booked_within_elapsed": (torch.relu(booked-elapsed-tolerance)/scale).square().mean(),
        }
