from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


def _gradient_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = prediction.new_tensor(0.0)
    for dim in (2, 3, 4):
        pred_diff = torch.diff(prediction, dim=dim)
        target_diff = torch.diff(target, dim=dim)
        loss = loss + F.l1_loss(pred_diff, target_diff)
    return loss / 3.0


def _ssim3d_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel = 5
    padding = kernel // 2
    # count_include_pad=True would average the zero padding into the border windows,
    # pulling every border mean toward zero and inventing structure that is not there.
    pool = partial(F.avg_pool3d, kernel_size=kernel, stride=1, padding=padding,
                   count_include_pad=False)
    mean_x = pool(prediction)
    mean_y = pool(target)
    # E[x^2] - E[x]^2 is only non-negative in exact arithmetic.
    var_x = (pool(prediction.square()) - mean_x.square()).clamp_min(0.0)
    var_y = (pool(target.square()) - mean_y.square()).clamp_min(0.0)
    covariance = pool(prediction * target) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2) + 1e-8
    )
    return 1.0 - score.clamp(-1, 1).mean()


class RestorationLoss(nn.Module):
    def __init__(
        self,
        artifact_l1: float = 5.0,
        global_l1: float = 1.0,
        gradient: float = 0.5,
        ssim: float = 0.5,
        identity: float = 2.0,
        uncertainty: float = 0.1,
    ) -> None:
        super().__init__()
        self.weights = {
            "artifact_l1": artifact_l1,
            "global_l1": global_l1,
            "gradient": gradient,
            "ssim": ssim,
            "identity": identity,
            "uncertainty": uncertainty,
        }

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        corrupted: torch.Tensor,
        artifact_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = outputs["corrected"]
        mask = artifact_mask.float().clamp(0, 1)
        inverse = 1.0 - mask
        absolute = (prediction - target).abs()
        artifact_l1 = (absolute * mask).sum() / mask.sum().clamp_min(1.0)
        global_l1 = absolute.mean()
        identity = ((prediction - corrupted).abs() * inverse).sum() / inverse.sum().clamp_min(1.0)
        gradient = _gradient_l1(prediction, target)
        ssim = _ssim3d_loss(prediction, target)
        error_sq = (prediction - target).square()
        log_var = outputs["log_variance"]
        uncertainty = (
            (torch.exp(-log_var) * error_sq + log_var) * mask
        ).sum() / mask.sum().clamp_min(1.0)
        mask_bce = F.binary_cross_entropy_with_logits(outputs["artifact_logit"], mask)
        terms = {
            "artifact_l1": artifact_l1,
            "global_l1": global_l1,
            "gradient": gradient,
            "ssim": ssim,
            "identity": identity,
            "uncertainty": uncertainty,
            "mask_bce": mask_bce,
        }
        total = sum(self.weights[name] * terms[name] for name in self.weights) + 0.1 * mask_bce
        return total, terms
