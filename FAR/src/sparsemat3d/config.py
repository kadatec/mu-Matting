"""TOML configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path

from .constants import SEQUENCE_LENGTH, TRAINING_DATASET_REPEATS

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


_PATH_KEYS = {
    ("model", "first_stage_config"),
    ("model", "first_stage_checkpoint"),
    ("data", "rvm_root"),
    ("data", "background_root"),
    ("data", "mattehuman_root"),
    ("data", "validation_manifest"),
    ("data", "evaluation_manifest"),
    ("output", "best_checkpoint"),
    ("output", "final_checkpoint"),
}
_DATA_FIELDS = {
    "sequence_length",
    "rvm_root",
    "background_root",
    "mattehuman_root",
    "validation_manifest",
    "evaluation_manifest",
    "rvm_repeat",
    "mattehuman_repeat",
    "image_size",
    "lowres_size",
}
_TRAIN_FIELDS = {
    "batch_size",
    "num_workers",
    "epochs",
    "learning_rate",
    "min_learning_rate",
    "betas",
    "decay_every",
    "validation_interval",
    "seed",
}
_LOADER_FIELDS = {"batch_size", "num_workers"}


def _require_fixed_integer(value: object, expected: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise ValueError(f"{field} is fixed to {expected}")


def _validate_image_size(value: object, field: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 1
            for dimension in value
        )
    ):
        raise ValueError(f"{field} must contain two positive integer dimensions")


def _require_integer(
    value: object,
    field: str,
    *,
    minimum: int,
) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ValueError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )


def _require_number(
    value: object,
    field: str,
    *,
    minimum: float,
) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"{field} must be greater than or equal to {minimum}")


def _reject_unknown_fields(
    section_name: str,
    values: dict,
    allowed_fields: set[str],
) -> None:
    unknown_fields = sorted(set(values) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            f"unknown {section_name} fields: {', '.join(unknown_fields)}"
        )


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    config = deepcopy(config)
    for section, key in _PATH_KEYS:
        value = config.get(section, {}).get(key)
        if value:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = config_path.parent / candidate
            config[section][key] = str(candidate.resolve())
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    data_config = config.get("data", {})
    _reject_unknown_fields("data", data_config, _DATA_FIELDS)
    _require_fixed_integer(
        data_config.get("sequence_length", SEQUENCE_LENGTH),
        SEQUENCE_LENGTH,
        "data.sequence_length",
    )
    for key in (
        "rvm_root",
        "background_root",
        "mattehuman_root",
        "validation_manifest",
        "evaluation_manifest",
    ):
        if not data_config.get(key):
            raise ValueError(f"data.{key} is required")
    configured_repeats = (
        data_config.get("rvm_repeat", TRAINING_DATASET_REPEATS[0]),
        data_config.get("mattehuman_repeat", TRAINING_DATASET_REPEATS[1]),
    )
    for field, value, expected in zip(
        ("data.rvm_repeat", "data.mattehuman_repeat"),
        configured_repeats,
        TRAINING_DATASET_REPEATS,
    ):
        _require_fixed_integer(value, expected, field)
    for field in ("image_size", "lowres_size"):
        _validate_image_size(data_config.get(field, [512, 512]), f"data.{field}")

    train_config = config.get("train", {})
    _reject_unknown_fields("train", train_config, _TRAIN_FIELDS)
    for field in ("batch_size", "epochs", "decay_every", "validation_interval"):
        _require_integer(train_config.get(field), f"train.{field}", minimum=1)
    _require_integer(
        train_config.get("num_workers"),
        "train.num_workers",
        minimum=0,
    )
    _require_integer(train_config.get("seed"), "train.seed", minimum=0)
    for field in ("learning_rate", "min_learning_rate"):
        _require_number(train_config.get(field), f"train.{field}", minimum=0.0)
    if train_config["min_learning_rate"] > train_config["learning_rate"]:
        raise ValueError(
            "train.min_learning_rate cannot exceed train.learning_rate"
        )
    betas = train_config.get("betas")
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(
            not isinstance(beta, (int, float))
            or isinstance(beta, bool)
            or not isfinite(beta)
            or beta < 0
            or beta >= 1
            for beta in betas
        )
    ):
        raise ValueError("train.betas must contain two values in [0, 1)")

    for section_name in ("validation", "evaluation"):
        section_config = config.get(section_name, {})
        _reject_unknown_fields(
            section_name,
            section_config,
            _LOADER_FIELDS,
        )
        _require_integer(
            section_config.get("batch_size"),
            f"{section_name}.batch_size",
            minimum=1,
        )
        _require_integer(
            section_config.get("num_workers"),
            f"{section_name}.num_workers",
            minimum=0,
        )

    model_config = config.get("model", {})
    for key in ("first_stage_config", "first_stage_checkpoint"):
        if not model_config.get(key):
            raise ValueError(f"model.{key} is required")
    output_config = config.get("output", {})
    for key in ("best_checkpoint", "final_checkpoint"):
        if not output_config.get(key):
            raise ValueError(f"output.{key} is required")
