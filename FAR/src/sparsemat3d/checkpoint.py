"""Joint model checkpoint I/O."""

from __future__ import annotations

import os
from pathlib import Path

import torch

CHECKPOINT_FORMAT = "sparsemat3d-joint-v1"


def save_joint_checkpoint(model: torch.nn.Module, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": model.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def load_joint_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    payload = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != CHECKPOINT_FORMAT
        or not isinstance(payload.get("state_dict"), dict)
    ):
        raise ValueError("not a supported FAR joint checkpoint")
    model.load_state_dict(payload["state_dict"], strict=True)
