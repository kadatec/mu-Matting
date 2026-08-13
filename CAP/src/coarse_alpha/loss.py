import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from torch import Tensor


class LaplacianPyramidLoss(nn.Module):
    """L1 distance over Laplacian pyramids with persistent kernels."""

    def __init__(self, max_levels: int = 5) -> None:
        super().__init__()
        self.max_levels = int(max_levels)
        kernel = torch.tensor(
            [
                [1.0, 4.0, 6.0, 4.0, 1.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [6.0, 24.0, 36.0, 24.0, 6.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [1.0, 4.0, 6.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        ).div_(256.0)
        self.register_buffer(
            'gaussian_kernel', kernel.unsqueeze(0).unsqueeze(0))
        self.register_buffer(
            'upsample_kernel',
            (kernel * 4.0).unsqueeze(0).unsqueeze(0),
        )

    def _filter(self, image: Tensor, upsample: bool = False) -> Tensor:
        channels = image.shape[1]
        base_kernel = (
            self.upsample_kernel if upsample else self.gaussian_kernel)
        kernel = base_kernel.expand(channels, 1, 5, 5)
        image = F.pad(image, (2, 2, 2, 2), mode='reflect')
        return F.conv2d(image, kernel, groups=channels)

    @staticmethod
    def _downsample(image: Tensor) -> Tensor:
        return image[:, :, ::2, ::2]

    def _upsample(self, image: Tensor) -> Tensor:
        upsampled = image.new_zeros(
            image.shape[0],
            image.shape[1],
            image.shape[2] * 2,
            image.shape[3] * 2,
        )
        upsampled[:, :, ::2, ::2] = image
        return self._filter(upsampled, upsample=True)

    def _pyramid(self, image: Tensor) -> list:
        pyramid = []
        current = image
        for _ in range(self.max_levels):
            filtered = self._filter(current)
            downsampled = self._downsample(filtered)
            upsampled = self._upsample(downsampled)
            pyramid.append(current - upsampled)
            current = downsampled
        return pyramid

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction_levels = self._pyramid(prediction)
        target_levels = self._pyramid(target)
        return sum(
            F.l1_loss(prediction_level, target_level)
            for prediction_level, target_level
            in zip(prediction_levels, target_levels)
        )


@MODELS.register_module()
class CoarseAlphaLoss(nn.Module):
    """Pixel L1 plus the coarse-alpha Laplacian objective."""

    def __init__(
        self,
        reduction: str = 'mean',
        loss_weight: float = 1.0,
        laplacian_weight: float = 1.0,
        max_levels: int = 5,
        loss_name: str = 'loss_alpha',
    ) -> None:
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Unsupported reduction {reduction!r}')
        self.reduction = reduction
        self.loss_weight = float(loss_weight)
        self.laplacian_weight = float(laplacian_weight)
        self._loss_name = loss_name
        self.laplacian = LaplacianPyramidLoss(max_levels=max_levels)

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        weight: Tensor = None,
        reduction_override: str = None,
        **kwargs,
    ) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f'Prediction {prediction.shape} and target {target.shape} '
                'must have identical shapes')
        reduction = reduction_override or self.reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Unsupported reduction {reduction!r}')

        pixel_loss = F.l1_loss(prediction, target, reduction='none')
        if weight is not None:
            pixel_loss = pixel_loss * weight
        if reduction == 'mean':
            pixel_loss = pixel_loss.mean()
        elif reduction == 'sum':
            pixel_loss = pixel_loss.sum()

        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(1)
            target = target.unsqueeze(1)
        if prediction.ndim != 4:
            raise ValueError('CoarseAlphaLoss expects CHW or NCHW tensors')

        laplacian_loss = self.laplacian(prediction, target)
        return self.loss_weight * (
            pixel_loss + self.laplacian_weight * laplacian_loss)

    @property
    def loss_name(self) -> str:
        return self._loss_name
