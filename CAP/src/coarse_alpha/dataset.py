import copy
import os
import pickle
from numbers import Integral
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS

IMAGE_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
PathLike = Union[str, os.PathLike]


def _list_images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _alpha_to_uint8(alpha: np.ndarray) -> np.ndarray:
    """Convert alpha values to uint8 with dtype-defined scaling."""
    if not isinstance(alpha, np.ndarray):
        raise TypeError('Alpha matte must be a NumPy array')
    if alpha.size == 0:
        raise ValueError('Alpha matte must not be empty')
    if alpha.dtype == np.uint8:
        return alpha
    if np.issubdtype(alpha.dtype, np.bool_):
        return alpha.astype(np.uint8) * 255
    if np.issubdtype(alpha.dtype, np.integer):
        if np.any(alpha < 0):
            minimum = int(alpha.min())
            raise ValueError(
                f'Integer alpha matte must be non-negative, got minimum '
                f'{minimum} for dtype {alpha.dtype}')
        dtype_maximum = np.iinfo(alpha.dtype).max
        scaled = alpha.astype(np.float64) * (255.0 / dtype_maximum)
        return np.rint(scaled).astype(np.uint8)
    if np.issubdtype(alpha.dtype, np.floating):
        if not np.isfinite(alpha).all():
            raise ValueError('Floating-point alpha matte must contain only '
                             'finite values')
        minimum = float(alpha.min())
        maximum = float(alpha.max())
        if minimum < 0.0:
            raise ValueError(
                f'Floating-point alpha matte must be non-negative, got '
                f'minimum {minimum}')
        if maximum <= 1.0:
            scaled = alpha.astype(np.float64) * 255.0
        elif maximum <= 255.0:
            scaled = alpha.astype(np.float64)
        else:
            raise ValueError(
                'Floating-point alpha matte values must be in [0, 1] or '
                f'[0, 255], got maximum {maximum}')
        return np.rint(scaled).astype(np.uint8)
    raise TypeError(f'Unsupported alpha matte dtype {alpha.dtype}')


def _read_image_and_alpha(
    image_path: str,
    alpha_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f'Unable to read image: {image_path}')

    embedded_alpha = None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        embedded_alpha = image[:, :, 3]
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f'Image must have 1, 3, or 4 channels: {image_path}')

    same_file = (
        alpha_path is not None
        and os.path.abspath(alpha_path) == os.path.abspath(image_path)
    )
    if same_file:
        alpha = embedded_alpha
    elif alpha_path is not None:
        alpha = cv2.imread(alpha_path, cv2.IMREAD_UNCHANGED)
        if alpha is None:
            raise FileNotFoundError(f'Unable to read alpha matte: {alpha_path}')
        if alpha.ndim == 3:
            alpha = alpha[:, :, 0]
    else:
        alpha = embedded_alpha

    if alpha is None:
        raise ValueError(f'No alpha matte found for {image_path!r}')
    alpha = _alpha_to_uint8(alpha)
    if alpha.shape != image.shape[:2]:
        raise ValueError(
            f'Image and alpha matte must have matching spatial shapes: '
            f'{image_path!r} has {image.shape[:2]}, '
            f'{alpha_path!r} has {alpha.shape}'
        )
    return image, alpha


@DATASETS.register_module()
class ImageAlphaDataset(BaseDataset):
    """Dataset of paired images and alpha mattes."""

    def __init__(
        self,
        roots: Sequence[PathLike],
        repeat_factors: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> None:
        if isinstance(roots, (str, os.PathLike)) or not isinstance(
                roots, Sequence):
            raise TypeError('roots must be a sequence of paths')
        self.roots = []
        for index, root in enumerate(roots):
            if not isinstance(root, (str, os.PathLike)):
                raise TypeError(f'roots[{index}] must be a path')
            root_string = os.fspath(root)
            if not root_string:
                raise ValueError(f'roots[{index}] must not be empty')
            self.roots.append(root_string)
        if not self.roots:
            raise ValueError('roots must contain at least one path')

        if repeat_factors is None:
            repeat_factors = [1] * len(self.roots)
        if isinstance(repeat_factors, (str, bytes)) or not isinstance(
                repeat_factors, Sequence):
            raise TypeError('repeat_factors must be a sequence of integers')
        if len(repeat_factors) != len(self.roots):
            raise ValueError(
                'repeat_factors must have one value for each root: '
                f'got {len(repeat_factors)} factors for {len(self.roots)} roots')
        self.repeat_factors = []
        for index, factor in enumerate(repeat_factors):
            if isinstance(factor, bool) or not isinstance(factor, Integral):
                raise TypeError(
                    f'repeat_factors[{index}] must be an integer')
            if factor < 1:
                raise ValueError(
                    f'repeat_factors[{index}] must be positive')
            self.repeat_factors.append(int(factor))

        super().__init__(data_root=self.roots[0], **kwargs)

    @staticmethod
    def _layout(root: Path) -> Tuple[Path, Path]:
        for image_name, alpha_name in (
            ('images', 'alphas'),
            ('fg', 'alpha'),
        ):
            image_directory = root / image_name
            alpha_directory = root / alpha_name
            if image_directory.is_dir() and alpha_directory.is_dir():
                return image_directory, alpha_directory
        raise FileNotFoundError(
            f'Dataset root {str(root)!r} must contain either '
            'images/ and alphas/ or fg/ and alpha/')

    def _load_root(self, root: str) -> List[dict]:
        root_path = Path(root)
        image_directory, alpha_directory = self._layout(root_path)
        alpha_by_stem = {}
        for alpha_path in _list_images(alpha_directory):
            if alpha_path.stem in alpha_by_stem:
                raise ValueError(
                    f'Duplicate alpha matte stem {alpha_path.stem!r} in '
                    f'{str(alpha_directory)!r}')
            alpha_by_stem[alpha_path.stem] = alpha_path

        data = []
        for image_path in _list_images(image_directory):
            alpha_path = alpha_by_stem.get(image_path.stem)
            if alpha_path is None and image_path.suffix.lower() == '.png':
                alpha_path = image_path
            if alpha_path is not None:
                data.append(dict(
                    img_path=str(image_path),
                    alpha_path=str(alpha_path),
                ))
        return data

    def load_data_list(self) -> List[dict]:
        data_list = []
        for root, repeat_factor in zip(self.roots, self.repeat_factors):
            root_items = self._load_root(root)
            if not root_items:
                raise ValueError(
                    f'Dataset root {root!r} contains no image-alpha pairs')
            data_list.extend(root_items * repeat_factor)
        return data_list

    def get_data_info(self, idx: int) -> dict:
        if self.serialize_data:
            start = 0 if idx == 0 else self.data_address[idx - 1].item()
            end = self.data_address[idx].item()
            data_info = pickle.loads(memoryview(self.data_bytes[start:end]))
        else:
            data_info = copy.deepcopy(self.data_list[idx])

        image, alpha = _read_image_and_alpha(
            data_info['img_path'], data_info['alpha_path'])
        data_info.update(
            img=image,
            alpha=alpha,
            img_shape=image.shape[:2],
            ori_shape=image.shape[:2],
        )
        return data_info
