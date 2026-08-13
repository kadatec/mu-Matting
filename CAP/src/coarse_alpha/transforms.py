import random
from numbers import Integral, Real
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import PixelData
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample

Size = Union[int, Sequence[int]]


def _normalize_size(size: Size, name: str) -> Tuple[int, int]:
    if isinstance(size, bool):
        raise TypeError(f'{name} must contain integers')
    if isinstance(size, Integral):
        height = width = int(size)
    else:
        if isinstance(size, (str, bytes)) or not isinstance(size, Sequence):
            raise TypeError(
                f'{name} must be an integer or a (height, width) sequence')
        if len(size) != 2:
            raise ValueError(
                f'{name} must contain exactly (height, width)')
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               for value in size):
            raise TypeError(f'{name} must contain integers')
        height, width = (int(value) for value in size)
    if height < 1 or width < 1:
        raise ValueError(
            f'{name} values must be positive, got {(height, width)}')
    return height, width


def _validate_image_alpha(results: dict) -> Tuple[np.ndarray, np.ndarray]:
    try:
        image = results['img']
        alpha = results['alpha']
    except KeyError as error:
        raise KeyError(
            f'Missing required transform input {error.args[0]!r}') from error
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError('img must be an HWC NumPy array')
    if not isinstance(alpha, np.ndarray) or alpha.ndim != 2:
        raise ValueError('alpha must be an HW NumPy array')
    if image.shape[:2] != alpha.shape:
        raise ValueError(
            'img and alpha must have matching spatial shapes, got '
            f'{image.shape[:2]} and {alpha.shape}')
    return image, alpha


@TRANSFORMS.register_module()
class RandomCropAndFlipPair(BaseTransform):
    """Randomly crop and horizontally flip an image-alpha pair."""

    def __init__(
        self,
        crop_size: Size,
        flip_probability: float = 0.5,
    ) -> None:
        self.crop_size = _normalize_size(crop_size, 'crop_size')
        if isinstance(flip_probability, bool) or not isinstance(
                flip_probability, Real):
            raise TypeError('flip_probability must be a real number')
        self.flip_probability = float(flip_probability)
        if not 0.0 <= self.flip_probability <= 1.0:
            raise ValueError('flip_probability must be between 0 and 1')

    def transform(self, results: dict) -> dict:
        image, alpha = _validate_image_alpha(results)
        crop_height, crop_width = self.crop_size
        height, width = image.shape[:2]
        if height < crop_height or width < crop_width:
            raise ValueError(
                f'crop_size {self.crop_size} exceeds image size '
                f'{(height, width)}')

        top = np.random.randint(0, height - crop_height + 1)
        left = np.random.randint(0, width - crop_width + 1)
        bottom = top + crop_height
        right = left + crop_width
        image = image[top:bottom, left:right]
        alpha = alpha[top:bottom, left:right]

        flipped = random.random() < self.flip_probability
        if flipped:
            image = image[:, ::-1].copy()
            alpha = alpha[:, ::-1].copy()

        results.update(
            img=image,
            alpha=alpha,
            img_shape=image.shape[:2],
            ori_shape=image.shape[:2],
            flip=flipped,
            flip_direction='horizontal' if flipped else None,
        )
        return results


@TRANSFORMS.register_module()
class ResizeImageAndAlpha(BaseTransform):
    """Resize an image-alpha pair to ``(height, width)``."""

    def __init__(self, size: Size) -> None:
        self.size = _normalize_size(size, 'size')

    def transform(self, results: dict) -> dict:
        image, alpha = _validate_image_alpha(results)
        height, width = self.size
        results['img'] = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_LINEAR)
        results['alpha'] = cv2.resize(
            alpha, (width, height), interpolation=cv2.INTER_LINEAR)
        results['img_shape'] = (height, width)
        results['ori_shape'] = (height, width)
        return results


@TRANSFORMS.register_module()
class PackAlphaMattingInputs(BaseTransform):
    """Pack an image and normalized alpha matte for the model."""

    def __init__(
        self,
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'scale_factor',
            'flip',
            'flip_direction',
        ),
    ) -> None:
        self.meta_keys = tuple(meta_keys)

    def transform(self, results: dict) -> dict:
        image, alpha = _validate_image_alpha(results)
        image = to_tensor(image.transpose(2, 0, 1)).contiguous()
        alpha = to_tensor(alpha).float().unsqueeze(0) / 255.0

        data_sample = SegDataSample()
        data_sample.set_data(dict(
            gt_depth_map=PixelData(data=alpha.contiguous())))
        data_sample.set_metainfo({
            key: results[key] for key in self.meta_keys if key in results
        })
        return dict(inputs=image, data_samples=data_sample)
