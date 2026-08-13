"""Four-frame data interfaces shared by train, evaluation, and inference."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .constants import SEQUENCE_LENGTH

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_TRAIN_CROP_SIZES = (512, 640, 800)


def _load_rgb_image(path: Path, size: tuple[int, int] | None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None:
            image = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)


def _load_alpha_image(path: Path, size: tuple[int, int] | None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        if size is not None:
            image = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(image).copy()).unsqueeze(0).float().div(255)


def _create_lowres_images(
    images: torch.Tensor, lowres_size: tuple[int, int]
) -> torch.Tensor:
    resized_images = F.interpolate(
        images.float(), size=lowres_size, mode="bilinear", align_corners=False
    ).round().clamp(0, 255).to(torch.uint8)
    return resized_images[:, [2, 1, 0]]


def _natural_sort_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.stem)
    ]


def _collect_image_paths_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    }


def _sample_window_starts(frame_count: int) -> list[int]:
    window_starts = list(
        range(
            0,
            frame_count - (SEQUENCE_LENGTH * 2 - 1),
            SEQUENCE_LENGTH * 2,
        )
    )
    return window_starts or [0]


class TrainingFrameSampler:
    """Sample one temporal perturbation shared by every clip modality."""

    playback_speeds = (0.5, 1, 2, 3, 4, 5)

    def __call__(self) -> list[int]:
        playback_speed = random.choice(self.playback_speeds)
        frame_indices = [
            int(index * playback_speed) for index in range(SEQUENCE_LENGTH)
        ]
        starting_offset = random.randint(0, SEQUENCE_LENGTH - 1)
        frame_indices = [
            frame_index + starting_offset for frame_index in frame_indices
        ]
        if random.random() < 0.5:
            frame_indices.reverse()
        return frame_indices


def _resize_rgb_image(
    image: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    if tuple(image.shape[-2:]) == size:
        return image
    return (
        F.interpolate(
            image.unsqueeze(0).float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )


def _load_rgb_images(paths: list[Path]) -> torch.Tensor:
    return torch.stack([_load_rgb_image(path, size=None) for path in paths])


def _load_alpha_images(paths: list[Path]) -> torch.Tensor:
    return torch.stack([_load_alpha_image(path, size=None) for path in paths])


def _augment_rvm_sample(
    foregrounds: torch.Tensor,
    backgrounds: torch.Tensor,
    alphas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply light clip-shared augmentation before online composition."""
    if random.random() < 0.5:
        foregrounds = foregrounds.flip(-1)
        backgrounds = backgrounds.flip(-1)
        alphas = alphas.flip(-1)
    if random.random() < 0.8:
        gain = random.uniform(0.9, 1.1)
        offset = random.uniform(-8.0, 8.0)
        saturation = random.uniform(0.9, 1.1)
        augmented_images = []
        for images in (foregrounds, backgrounds):
            values = images.float()
            gray = values.mean(dim=1, keepdim=True)
            values = gray + saturation * (values - gray)
            augmented_images.append(
                (values * gain + offset).round().clamp(0, 255).to(torch.uint8)
            )
        foregrounds, backgrounds = augmented_images
    if random.random() < 0.2:
        foregrounds = (
            F.avg_pool2d(
                foregrounds.float(), kernel_size=3, stride=1, padding=1
            )
            .round()
            .to(torch.uint8)
        )
        backgrounds = (
            F.avg_pool2d(
                backgrounds.float(), kernel_size=3, stride=1, padding=1
            )
            .round()
            .to(torch.uint8)
        )
    return foregrounds, backgrounds, alphas


def _crop_and_resize_sample(
    image_tensors: dict[str, torch.Tensor],
    alphas: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Apply one shared alpha-guided crop and return normalized frame boxes."""
    height, width = alphas.shape[-2:]
    requested_crop_size = random.choice(_TRAIN_CROP_SIZES)
    crop_height = min(requested_crop_size, height)
    crop_width = min(requested_crop_size, width)
    mean_alpha = alphas.mean(dim=(0, 1))
    candidate_pixels = (
        (mean_alpha > 1 / 255) & (mean_alpha < 254 / 255)
    ).nonzero()
    if not len(candidate_pixels):
        candidate_pixels = (mean_alpha > 1 / 255).nonzero()
    if len(candidate_pixels):
        selected_pixel = candidate_pixels[random.randrange(len(candidate_pixels))]
        center_y, center_x = selected_pixel.tolist()
    else:
        center_y = random.randrange(height)
        center_x = random.randrange(width)
    top = max(0, min(height - crop_height, center_y - crop_height // 2))
    left = max(0, min(width - crop_width, center_x - crop_width // 2))
    crop_slices = (
        ...,
        slice(top, top + crop_height),
        slice(left, left + crop_width),
    )
    normalized_crop_box = alphas.new_tensor(
        (
            left / width,
            top / height,
            (left + crop_width) / width,
            (top + crop_height) / height,
        )
    )
    crop_boxes = normalized_crop_box.repeat(alphas.shape[0], 1)
    cropped_alphas = alphas[crop_slices]
    cropped_images = {
        key: value[crop_slices] for key, value in image_tensors.items()
    }
    if (crop_height, crop_width) != image_size:
        cropped_alphas = F.interpolate(
            cropped_alphas,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        cropped_images = {
            key: F.interpolate(
                value.float(), size=image_size, mode="bilinear", align_corners=False
            )
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            for key, value in cropped_images.items()
        }
    return cropped_images, cropped_alphas, crop_boxes


def _normalize_rgb_images(images: torch.Tensor) -> torch.Tensor:
    return (images.float().div(127.5) - 1).permute(1, 0, 2, 3)


def _build_training_sample(
    images: torch.Tensor,
    alphas: torch.Tensor,
    foregrounds: torch.Tensor,
    backgrounds: torch.Tensor,
    *,
    image_size: tuple[int, int],
    lowres_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    lowres_images = _create_lowres_images(images, lowres_size)
    cropped_images, cropped_alphas, crop_boxes = _crop_and_resize_sample(
        {
            "images": images,
            "foregrounds": foregrounds,
            "backgrounds": backgrounds,
        },
        alphas,
        image_size,
    )
    return {
        "images": _normalize_rgb_images(cropped_images["images"]),
        "lowres_images": lowres_images,
        "crop_boxes": crop_boxes,
        "alphas": cropped_alphas.permute(1, 0, 2, 3),
        "foregrounds": _normalize_rgb_images(cropped_images["foregrounds"]),
        "backgrounds": _normalize_rgb_images(cropped_images["backgrounds"]),
    }


class RVMVideoMattingDataset(Dataset):
    """Four-frame RVM clips composited on randomly sampled background videos."""

    def __init__(
        self,
        dataset_root: str | Path,
        background_root: str | Path,
        *,
        image_size: tuple[int, int] = (512, 512),
        lowres_size: tuple[int, int] = (512, 512),
        apply_augmentation: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.background_root = Path(background_root).expanduser().resolve()
        self.image_size = image_size
        self.lowres_size = lowres_size
        self.apply_augmentation = apply_augmentation
        self.frame_sampler = TrainingFrameSampler()
        self.frame_sequences: list[list[tuple[Path, Path]]] = []
        foreground_directory = self.dataset_root / "fgr"
        alpha_directory = self.dataset_root / "pha"
        if not foreground_directory.is_dir() or not alpha_directory.is_dir():
            raise FileNotFoundError("RVM root must contain fgr/ and pha/")
        for clip_directory in sorted(
            (path for path in foreground_directory.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            foreground_paths = _collect_image_paths_by_stem(clip_directory)
            alpha_paths = _collect_image_paths_by_stem(
                alpha_directory / clip_directory.name
            )
            shared_frame_names = sorted(
                foreground_paths.keys() & alpha_paths.keys(),
                key=lambda name: _natural_sort_key(Path(name)),
            )
            if shared_frame_names:
                self.frame_sequences.append(
                    [
                        (foreground_paths[name], alpha_paths[name])
                        for name in shared_frame_names
                    ]
                )
        self.background_sequences = [
            sorted(
                _collect_image_paths_by_stem(path).values(),
                key=_natural_sort_key,
            )
            for path in sorted(
                (item for item in self.background_root.iterdir() if item.is_dir()),
                key=lambda item: item.name,
            )
        ] if self.background_root.is_dir() else []
        self.background_sequences = [
            sequence for sequence in self.background_sequences if sequence
        ]
        if not self.frame_sequences:
            raise ValueError("RVM root contains no matched fgr/pha clips")
        if not self.background_sequences:
            raise ValueError("background root contains no video clips")
        self.sample_windows = [
            (sequence_index, window_start)
            for sequence_index, sequence in enumerate(self.frame_sequences)
            for window_start in _sample_window_starts(len(sequence))
        ]

    def __len__(self) -> int:
        return len(self.sample_windows)

    def __getitem__(self, index: int):
        sequence_index, window_start = self.sample_windows[index]
        frame_records = self.frame_sequences[sequence_index]
        sampled_offsets = self.frame_sampler()
        selected_records = [
            frame_records[
                (window_start + sampled_offset * 2) % len(frame_records)
            ]
            for sampled_offset in sampled_offsets
        ]
        foregrounds = _load_rgb_images(
            [record[0] for record in selected_records]
        )
        alphas = _load_alpha_images([record[1] for record in selected_records])
        background_sequence = random.choice(self.background_sequences)
        background_start = random.randrange(len(background_sequence))
        background_paths = [
            background_sequence[
                (background_start + sampled_offset) % len(background_sequence)
            ]
            for sampled_offset in sampled_offsets
        ]
        backgrounds = torch.stack(
            [
                _resize_rgb_image(
                    _load_rgb_image(path, size=None),
                    tuple(foregrounds.shape[-2:]),
                )
                for path in background_paths
            ]
        )
        if self.apply_augmentation:
            foregrounds, backgrounds, alphas = _augment_rvm_sample(
                foregrounds, backgrounds, alphas
            )
        images = (
            foregrounds.float() * alphas
            + backgrounds.float() * (1 - alphas)
        ).round().clamp(0, 255).to(torch.uint8)
        return _build_training_sample(
            images,
            alphas,
            foregrounds,
            backgrounds,
            image_size=self.image_size,
            lowres_size=self.lowres_size,
        )


class MatteHumanVideoMattingDataset(Dataset):
    """Read MatteHuman's existing composites and composition-loss sources."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        image_size: tuple[int, int] = (512, 512),
        lowres_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.image_size = image_size
        self.lowres_size = lowres_size
        self.frame_sampler = TrainingFrameSampler()
        modality_directories = {
            "image": self.dataset_root / "img",
            "alpha": self.dataset_root / "pha",
            "foreground": self.dataset_root / "fgr",
            "background": self.dataset_root / "bgr",
        }
        if not all(path.is_dir() for path in modality_directories.values()):
            raise FileNotFoundError("MatteHuman root must contain img/, pha/, fgr/, and bgr/")
        self.frame_sequences: list[list[dict[str, Path]]] = []
        for clip_directory in sorted(
            (
                path
                for path in modality_directories["image"].iterdir()
                if path.is_dir()
            ),
            key=lambda path: path.name,
        ):
            modality_paths = {
                modality: _collect_image_paths_by_stem(
                    directory / clip_directory.name
                )
                for modality, directory in modality_directories.items()
            }
            shared_frame_names = set.intersection(
                *(set(paths) for paths in modality_paths.values())
            )
            ordered_frame_names = sorted(
                shared_frame_names,
                key=lambda name: _natural_sort_key(Path(name)),
            )
            if ordered_frame_names:
                self.frame_sequences.append(
                    [
                        {
                            modality: paths[frame_name]
                            for modality, paths in modality_paths.items()
                        }
                        for frame_name in ordered_frame_names
                    ]
                )
        if not self.frame_sequences:
            raise ValueError("MatteHuman root contains no matched four-modality clips")
        self.sample_windows = [
            (sequence_index, window_start)
            for sequence_index, sequence in enumerate(self.frame_sequences)
            for window_start in _sample_window_starts(len(sequence))
        ]

    def __len__(self) -> int:
        return len(self.sample_windows)

    def __getitem__(self, index: int):
        sequence_index, window_start = self.sample_windows[index]
        frame_records = self.frame_sequences[sequence_index]
        sampled_offsets = self.frame_sampler()
        selected_records = [
            frame_records[
                (window_start + sampled_offset * 2) % len(frame_records)
            ]
            for sampled_offset in sampled_offsets
        ]
        images = _load_rgb_images(
            [record["image"] for record in selected_records]
        )
        alphas = _load_alpha_images(
            [record["alpha"] for record in selected_records]
        )
        foregrounds = _load_rgb_images(
            [record["foreground"] for record in selected_records]
        )
        backgrounds = _load_rgb_images(
            [record["background"] for record in selected_records]
        )
        return _build_training_sample(
            images,
            alphas,
            foregrounds,
            backgrounds,
            image_size=self.image_size,
            lowres_size=self.lowres_size,
        )


def save_alpha_png(alpha: torch.Tensor, output_path: str | Path) -> Path:
    """Save one two-dimensional alpha tensor in [0, 1] as an 8-bit PNG."""
    if alpha.ndim != 2:
        raise ValueError("alpha must be a two-dimensional tensor")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.round(
        alpha.detach().cpu().clamp(0, 1).numpy() * 255
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(destination, format="PNG")
    return destination


class FourFrameMattingDataset(Dataset):
    """Read four-frame clips from a small JSON manifest.

    The manifest is either a list or ``{"clips": [...]}``. Each clip contains
    exactly four frame objects with ``image`` and ``alpha`` paths. ``foreground``
    and ``background`` are required only when composition loss is enabled.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        image_size: tuple[int, int] | None = None,
        lowres_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.manifest_path = Path(manifest).expanduser().resolve()
        with self.manifest_path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        self.frame_sequences = (
            document["clips"] if isinstance(document, dict) else document
        )
        if not isinstance(self.frame_sequences, list):
            raise TypeError("manifest must contain a list of clips")
        self.image_size = image_size
        self.lowres_size = lowres_size
        for sequence_index, sequence in enumerate(self.frame_sequences):
            if not isinstance(sequence, dict):
                raise TypeError(f"clip {sequence_index} must be an object")
            frame_records = sequence.get("frames", [])
            if len(frame_records) != SEQUENCE_LENGTH:
                raise ValueError(
                    f"clip {sequence_index} must contain exactly "
                    f"{SEQUENCE_LENGTH} frames"
                )
            for frame_index, frame_record in enumerate(frame_records):
                if not isinstance(frame_record, dict):
                    raise TypeError(
                        f"clip {sequence_index} frame {frame_index} "
                        "must be an object"
                    )
                if not {"image", "alpha"} <= frame_record.keys():
                    raise ValueError("every frame needs image and alpha paths")
                composition_fields = {
                    "foreground",
                    "background",
                } & frame_record.keys()
                if composition_fields and len(composition_fields) != 2:
                    raise ValueError(
                        "foreground and background paths must be provided together"
                    )
                for field in ("image", "alpha", *sorted(composition_fields)):
                    self._resolve_frame_path(frame_record[field])

    def __len__(self) -> int:
        return len(self.frame_sequences)

    def _resolve_frame_path(self, value: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError("manifest frame paths must be non-empty strings")
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.manifest_path.parent / path

    def __getitem__(self, index: int):
        frame_records = self.frame_sequences[index]["frames"]
        images = torch.stack(
            [
                _load_rgb_image(
                    self._resolve_frame_path(frame["image"]),
                    self.image_size,
                )
                for frame in frame_records
            ]
        )
        sample = {
            "images": _normalize_rgb_images(images),
            "lowres_images": _create_lowres_images(
                images, self.lowres_size
            ),
            "alphas": torch.stack(
                [
                    _load_alpha_image(
                        self._resolve_frame_path(frame["alpha"]),
                        self.image_size,
                    )
                    for frame in frame_records
                ],
                dim=1,
            ),
        }
        if all(
            "foreground" in frame and "background" in frame
            for frame in frame_records
        ):
            for output_key, manifest_key in (
                ("foregrounds", "foreground"),
                ("backgrounds", "background"),
            ):
                source_images = torch.stack(
                    [
                        _load_rgb_image(
                            self._resolve_frame_path(frame[manifest_key]),
                            self.image_size,
                        )
                        for frame in frame_records
                    ]
                )
                sample[output_key] = _normalize_rgb_images(source_images)
        return sample


class VideoInferenceDataset(Dataset):
    """Read a video as non-overlapping four-frame clips, padding the last clip."""

    def __init__(
        self, video_path: str | Path, lowres_size: tuple[int, int] = (512, 512)
    ) -> None:
        self.video_path = Path(video_path).expanduser().resolve()
        self.lowres_size = lowres_size
        if not self.video_path.is_file():
            raise FileNotFoundError(f"video does not exist: {self.video_path}")
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            if not capture.isOpened():
                raise ValueError(f"cannot open video: {self.video_path}")
            ok, _ = capture.read()
            if not ok:
                raise ValueError(f"video contains no decodable frames: {self.video_path}")
            reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_count = max(1, reported)
        finally:
            capture.release()

    def __len__(self) -> int:
        return math.ceil(self.frame_count / SEQUENCE_LENGTH)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        capture = cv2.VideoCapture(str(self.video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index * SEQUENCE_LENGTH)
        decoded_frames = []
        try:
            for _ in range(SEQUENCE_LENGTH):
                ok, frame = capture.read()
                if not ok:
                    break
                decoded_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if not decoded_frames:
            raise RuntimeError(f"failed to decode clip {index} from {self.video_path}")
        decoded_frames.extend(
            [decoded_frames[-1]]
            * (SEQUENCE_LENGTH - len(decoded_frames))
        )
        images = torch.from_numpy(np.stack(decoded_frames)).permute(0, 3, 1, 2)
        return {
            "images": _normalize_rgb_images(images),
            "lowres_images": _create_lowres_images(
                images, self.lowres_size
            ),
            "valid_frame_count": min(
                SEQUENCE_LENGTH, self.frame_count - index * SEQUENCE_LENGTH
            ),
        }
