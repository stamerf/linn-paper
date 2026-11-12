import torch


def masked_mse(prediction, targets, mask, scale):
    if prediction.shape != targets.shape or mask.shape != targets.shape or mask.dtype != torch.bool:
        raise ValueError("Prediction, target and boolean mask shapes must match")
    if not torch.isfinite(prediction).all() or not torch.isfinite(targets[mask]).all():
        raise ValueError("Nonfinite prediction or observed target")
    scales = torch.as_tensor(scale, dtype=prediction.dtype, device=prediction.device)
    if scales.shape != (prediction.shape[1],) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Invalid target scales")
    if not mask.any():
        return prediction.sum()*0
    # Index after masking; no arithmetic on missing targets, including NaN placeholders.
    residual = (prediction-torch.where(mask, targets, prediction.detach()))/scales
    return residual[mask].square().mean()
