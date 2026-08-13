"""Shared CLI construction helpers."""

from __future__ import annotations

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .constants import TRAINING_DATASET_REPEATS
from .data import (
    FourFrameMattingDataset,
    MatteHumanVideoMattingDataset,
    RVMVideoMattingDataset,
)
from .metrics import MetricAccumulator


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def create_training_dataset(config: dict) -> ConcatDataset:
    data_config = config["data"]
    image_size = tuple(data_config.get("image_size", [512, 512]))
    lowres_size = tuple(data_config.get("lowres_size", [512, 512]))
    rvm_dataset = RVMVideoMattingDataset(
        data_config["rvm_root"],
        data_config["background_root"],
        image_size=image_size,
        lowres_size=lowres_size,
    )
    mattehuman_dataset = MatteHumanVideoMattingDataset(
        data_config["mattehuman_root"],
        image_size=image_size,
        lowres_size=lowres_size,
    )
    rvm_repeats, mattehuman_repeats = TRAINING_DATASET_REPEATS
    return ConcatDataset(
        [rvm_dataset] * rvm_repeats
        + [mattehuman_dataset] * mattehuman_repeats
    )


def create_matting_data_loader(
    config: dict,
    split: str,
    *,
    shuffle: bool,
) -> DataLoader:
    data_config = config["data"]
    if split == "train":
        dataset = create_training_dataset(config)
    elif split in {"validation", "evaluation"}:
        image_size = data_config.get("image_size")
        lowres_size = data_config.get("lowres_size", [512, 512])
        dataset = FourFrameMattingDataset(
            data_config[f"{split}_manifest"],
            image_size=tuple(image_size) if image_size else None,
            lowres_size=tuple(lowres_size),
        )
        if len(dataset) == 0:
            raise ValueError(f"{split} manifest contains no clips")
    else:
        raise ValueError(f"unsupported matting split: {split}")
    section = config[split]
    return DataLoader(
        dataset,
        batch_size=section.get("batch_size", 1),
        shuffle=shuffle,
        num_workers=section.get("num_workers", 0),
        pin_memory=True,
        drop_last=split == "train",
    )


def evaluate_matting_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    metrics = MetricAccumulator()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in data_loader:
                batch = move_batch_to_device(batch, device)
                model_output = model(batch)
                metrics.update(
                    model_output["refined_alphas"],
                    batch["alphas"],
                )
    finally:
        model.train(was_training)
    return metrics.compute()
