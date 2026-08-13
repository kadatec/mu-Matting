"""Losses for sparse video alpha refinement."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _create_gaussian_kernel(
    dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    kernel = torch.tensor(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        dtype=dtype,
        device=device,
    )
    return kernel.div(256).view(1, 1, 5, 5)


def _apply_gaussian_blur(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    kernel = _create_gaussian_kernel(
        image.dtype, image.device
    ).expand(channels, 1, 5, 5)
    padded_image = F.pad(image, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(padded_image, kernel, groups=channels)


def _build_laplacian_pyramid(
    image: torch.Tensor, levels: int
) -> list[torch.Tensor]:
    current_level = image
    pyramid_levels = []
    for _ in range(levels):
        if min(current_level.shape[-2:]) <= 2:
            break
        downsampled = _apply_gaussian_blur(current_level)[:, :, ::2, ::2]
        upsampled = F.interpolate(
            downsampled,
            size=current_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pyramid_levels.append(
            current_level - _apply_gaussian_blur(upsampled)
        )
        current_level = downsampled
    if not pyramid_levels:
        pyramid_levels.append(current_level)
    return pyramid_levels


class LaplacianPyramidLoss(nn.Module):
    """Channel-agnostic Laplacian pyramid loss."""

    def __init__(self, max_levels: int = 3) -> None:
        super().__init__()
        if max_levels < 1:
            raise ValueError("max_levels must be positive")
        self.max_levels = max_levels

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")
        return sum(
            F.l1_loss(a, b)
            for a, b in zip(
                _build_laplacian_pyramid(prediction, self.max_levels),
                _build_laplacian_pyramid(target, self.max_levels),
            )
        )


def alpha_matting_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    region_mask: torch.Tensor | None = None,
    laplacian_loss: LaplacianPyramidLoss | None = None,
):
    error = torch.sqrt((prediction - target).square() + 1e-10)
    if region_mask is None:
        loss = error.mean()
    else:
        loss = (
            (error * region_mask).sum()
            / region_mask.sum().clamp_min(1)
        )
    if laplacian_loss is not None:
        loss = loss + laplacian_loss(prediction, target)
    return loss


def _flatten_video_frames(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError("expected a B,C,T,H,W tensor")
    return tensor.permute(0, 2, 1, 3, 4).flatten(0, 1)


def video_matting_losses(
    multiscale_alphas: Sequence[torch.Tensor],
    batch: dict[str, torch.Tensor],
    unknown_region_mask: torch.Tensor,
    alpha_weights: Sequence[float] = (0.1, 0.1, 1.0),
    composition_weight: float = 0.0,
    coherence_weight: float = 0.0,
    laplacian_levels: int = 3,
):
    """Compute the three-scale video matting objective."""
    if len(multiscale_alphas) != len(alpha_weights):
        raise ValueError("alpha_weights must match the prediction scales")
    target_alphas = batch["alphas"]
    full_resolution = target_alphas.shape[-2:]
    flattened_target_alphas = _flatten_video_frames(target_alphas)
    flattened_unknown_region_mask = _flatten_video_frames(
        unknown_region_mask
    )
    laplacian_loss = LaplacianPyramidLoss(laplacian_levels)

    alpha_loss = target_alphas.new_zeros(())
    for predicted_alphas, weight in zip(multiscale_alphas, alpha_weights):
        flattened_prediction = F.interpolate(
            _flatten_video_frames(predicted_alphas),
            size=full_resolution,
            mode="bilinear",
            align_corners=False,
        )
        alpha_loss = alpha_loss + float(weight) * alpha_matting_loss(
            flattened_prediction,
            flattened_target_alphas,
            flattened_unknown_region_mask,
            laplacian_loss,
        )

    losses = {"alpha_loss": alpha_loss}
    final_alphas = F.interpolate(
        _flatten_video_frames(multiscale_alphas[-1]),
        size=full_resolution,
        mode="bilinear",
        align_corners=False,
    )
    if composition_weight:
        images = _flatten_video_frames(batch["images"])
        foregrounds = _flatten_video_frames(batch["foregrounds"])
        backgrounds = _flatten_video_frames(batch["backgrounds"])
        composites = (
            final_alphas * foregrounds
            + (1 - final_alphas) * backgrounds
        )
        composition_error = (
            torch.sqrt((composites - images).square() + 1e-12)
            * flattened_unknown_region_mask
        )
        losses["composition_loss"] = (
            float(composition_weight)
            * composition_error.sum()
            / (
                flattened_unknown_region_mask.sum().clamp_min(1)
                * images.shape[1]
            )
        )

    if coherence_weight:
        shared_unknown_regions = (
            unknown_region_mask[:, :, 1:].bool()
            & unknown_region_mask[:, :, :-1].bool()
        )
        temporal_error = (
            multiscale_alphas[-1][:, :, 1:]
            - multiscale_alphas[-1][:, :, :-1]
            - target_alphas[:, :, 1:]
            + target_alphas[:, :, :-1]
        ).square()
        losses["coherence_loss"] = (
            float(coherence_weight)
            * (temporal_error * shared_unknown_regions).sum()
            / shared_unknown_regions.sum().clamp_min(1)
        )

    losses["total_loss"] = sum(losses.values())
    return losses
