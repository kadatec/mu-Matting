from typing import Sequence

import cv2
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def _pixel_data(data_sample, key):
    if isinstance(data_sample, dict):
        return data_sample[key]['data']
    return getattr(data_sample, key).data


class GradientError:
    def __init__(self, sigma: float = 1.4) -> None:
        self.filter_x, self.filter_y = self._filters(sigma)

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        prediction_gradient = self._gradient(prediction)
        target_gradient = self._gradient(target)
        return float(np.square(
            target_gradient - prediction_gradient).sum() / 1000.0)

    def _gradient(self, image: np.ndarray) -> np.ndarray:
        filtered_x = cv2.filter2D(
            image, -1, self.filter_x, borderType=cv2.BORDER_REPLICATE)
        filtered_y = cv2.filter2D(
            image, -1, self.filter_y, borderType=cv2.BORDER_REPLICATE)
        return np.sqrt(filtered_x ** 2 + filtered_y ** 2)

    @staticmethod
    def _filters(sigma: float, epsilon: float = 0.01):
        half_size = np.ceil(
            sigma * np.sqrt(
                -2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = int(2 * half_size + 1)
        coordinates = np.arange(size, dtype=np.float64) - half_size
        gaussian = np.exp(
            -(coordinates ** 2) / (2 * sigma ** 2)
        ) / (sigma * np.sqrt(2 * np.pi))
        derivative = -coordinates * gaussian / sigma ** 2
        filter_x = np.outer(gaussian, derivative)
        filter_x /= np.sqrt(np.square(filter_x).sum())
        return filter_x, filter_x.T


class ConnectivityError:
    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        round_down = -np.ones_like(target)
        thresholds = np.arange(0.0, 1.1, 0.1)
        for index in range(1, len(thresholds)):
            intersection = (
                (target >= thresholds[index])
                & (prediction >= thresholds[index])
            ).astype(np.uint8)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(
                intersection, connectivity=4)
            omega = np.zeros_like(target)
            component_sizes = stats[1:, -1]
            if component_sizes.size:
                largest = int(np.argmax(component_sizes)) + 1
                omega[labels == largest] = 1
            update = (round_down == -1) & (omega == 0)
            round_down[update] = thresholds[index - 1]
        round_down[round_down == -1] = 1
        target_difference = target - round_down
        prediction_difference = prediction - round_down
        target_phi = 1 - target_difference * (target_difference >= 0.15)
        prediction_phi = (
            1 - prediction_difference * (prediction_difference >= 0.15))
        return float(
            np.abs(target_phi - prediction_phi).sum() / 1000.0)


@METRICS.register_module()
class AlphaMattingMetric(BaseMetric):
    """Evaluate alpha MAD, MSE, gradient, and connectivity errors."""

    def __init__(
        self,
        collect_device: str = 'cpu',
        prefix=None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.gradient_error = GradientError()
        self.connectivity_error = ConnectivityError()

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            prediction = _to_numpy(
                _pixel_data(data_sample, 'pred_depth_map')).squeeze()
            target = _to_numpy(
                _pixel_data(data_sample, 'gt_depth_map')).squeeze()
            prediction = np.clip(prediction, 0.0, 1.0)
            target = np.clip(target, 0.0, 1.0)
            if prediction.shape != target.shape:
                prediction = cv2.resize(
                    prediction,
                    (target.shape[1], target.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            difference = prediction - target
            self.results.append(dict(
                mad=float(np.abs(difference).mean() * 1000.0),
                mse=float(np.square(difference).mean() * 1000.0),
                grad=self.gradient_error(prediction, target),
                conn=self.connectivity_error(prediction, target),
            ))

    def compute_metrics(self, results: list) -> dict:
        if not results:
            return {name: 0.0 for name in ('mad', 'mse', 'grad', 'conn')}
        return {
            name: float(np.mean([row[name] for row in results]))
            for name in ('mad', 'mse', 'grad', 'conn')
        }
