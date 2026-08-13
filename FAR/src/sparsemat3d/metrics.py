"""FAR video evaluation metrics."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def compute_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(MAD, MSE)`` in that order, both scaled by 1e3."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    difference = prediction - target
    mad = difference.abs().mean() * 1e3
    mse = difference.square().mean() * 1e3
    return mad, mse


def temporal_dtssd(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Temporal derivative error, scaled by 1e2."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("expected equal B,C,T,H,W tensors")
    if prediction.shape[2] < 2:
        raise ValueError("temporal_dtssd needs at least two frames")
    error = (
        prediction[:, :, 1:]
        - prediction[:, :, :-1]
        - target[:, :, 1:]
        + target[:, :, :-1]
    )
    return error.square().mean().sqrt() * 1e2


class GradientError:
    def __init__(self, sigma: float = 1.4) -> None:
        self.filter_x, self.filter_y = self._filters(sigma)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError("expected equal B,C,T,H,W tensors")
        if prediction.shape[1] != 1:
            raise ValueError("gradient error expects single-channel alpha mattes")

        prediction_array = prediction.detach().cpu().float().numpy()
        target_array = target.detach().cpu().float().numpy()
        sample_errors = []
        for sample_prediction, sample_target in zip(
            prediction_array,
            target_array,
        ):
            total_error = 0.0
            for frame_index in range(sample_prediction.shape[1]):
                predicted_frame = sample_prediction[0, frame_index]
                target_frame = sample_target[0, frame_index]
                predicted_normalized = np.zeros_like(predicted_frame)
                target_normalized = np.zeros_like(target_frame)
                cv2.normalize(
                    predicted_frame,
                    predicted_normalized,
                    1.0,
                    0.0,
                    cv2.NORM_MINMAX,
                )
                cv2.normalize(
                    target_frame,
                    target_normalized,
                    1.0,
                    0.0,
                    cv2.NORM_MINMAX,
                )
                predicted_gradient = self._gradient(predicted_normalized)
                target_gradient = self._gradient(target_normalized)
                total_error += np.square(
                    target_gradient - predicted_gradient
                ).sum()
            sample_errors.append(
                total_error / (sample_prediction.shape[1] * 1000.0)
            )
        return float(np.mean(sample_errors))

    def _gradient(self, image: np.ndarray) -> np.ndarray:
        filtered_x = cv2.filter2D(
            image,
            -1,
            self.filter_x,
            borderType=cv2.BORDER_REPLICATE,
        )
        filtered_y = cv2.filter2D(
            image,
            -1,
            self.filter_y,
            borderType=cv2.BORDER_REPLICATE,
        )
        return np.sqrt(filtered_x**2 + filtered_y**2).astype(np.float32)

    @staticmethod
    def _filters(
        sigma: float,
        epsilon: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        half_size = np.ceil(
            sigma
            * np.sqrt(
                -2
                * np.log(
                    np.sqrt(2 * np.pi) * sigma * epsilon
                )
            )
        )
        size = int(2 * half_size + 1)
        coordinates = np.arange(size, dtype=np.float64) - half_size
        gaussian = np.exp(
            -(coordinates**2) / (2 * sigma**2)
        ) / (sigma * np.sqrt(2 * np.pi))
        derivative = -coordinates * gaussian / sigma**2
        filter_x = np.outer(gaussian, derivative)
        filter_x /= np.sqrt(np.square(filter_x).sum())
        return filter_x, filter_x.T


class MetricAccumulator:
    """Accumulate metrics without silently producing NaN on empty input."""

    def __init__(self) -> None:
        self._gradient_error = GradientError()
        self._batch_metrics: list[tuple[float, float, float, float]] = []

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        mad, mse = compute_metrics(prediction, target)
        grad = self._gradient_error(prediction, target)
        dtssd = temporal_dtssd(prediction, target)
        self._batch_metrics.append(
            (mad.item(), mse.item(), grad, dtssd.item())
        )

    def compute(self) -> dict[str, float]:
        if not self._batch_metrics:
            raise RuntimeError("cannot compute metrics: evaluation produced no batches")
        batch_count = len(self._batch_metrics)
        return {
            "mad": (
                sum(metrics[0] for metrics in self._batch_metrics)
                / batch_count
            ),
            "mse": (
                sum(metrics[1] for metrics in self._batch_metrics)
                / batch_count
            ),
            "grad": (
                sum(metrics[2] for metrics in self._batch_metrics)
                / batch_count
            ),
            "dtssd": (
                sum(metrics[3] for metrics in self._batch_metrics)
                / batch_count
            ),
        }
