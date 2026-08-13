"""FAR training CLI."""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from .checkpoint import save_joint_checkpoint
from .config import load_config
from .losses import video_matting_losses
from .model import build_model
from .runtime import (
    create_matting_data_loader,
    evaluate_matting_model,
    move_batch_to_device,
    resolve_device,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train FAR")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    train_config = config["train"]
    seed = train_config.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = resolve_device(args.device)
    composition_weight = config["loss"].get("composition_weight", 0.0)
    training_loader = create_matting_data_loader(
        config,
        "train",
        shuffle=True,
    )
    validation_loader = create_matting_data_loader(
        config,
        "validation",
        shuffle=False,
    )
    model = build_model(config, load_first_stage_weights=True).to(device)
    initial_learning_rate = train_config["learning_rate"]
    minimum_learning_rate = train_config["min_learning_rate"]
    optimizer = torch.optim.Adam(
        model.refiner.parameters(),
        lr=initial_learning_rate,
        betas=tuple(train_config.get("betas", [0.9, 0.999])),
    )
    decay_every = train_config["decay_every"]
    validation_interval = train_config["validation_interval"]
    best_mad = float("inf")

    for epoch in range(train_config["epochs"]):
        model.train()
        cumulative_loss = 0.0
        for batch in training_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            model_output = model(batch)
            losses = video_matting_losses(
                model_output["multiscale_alphas"],
                batch,
                model_output["unknown_region_mask"],
                alpha_weights=config["loss"].get(
                    "alpha_weights", [0.1, 0.1, 1.0]
                ),
                composition_weight=composition_weight,
                coherence_weight=config["loss"].get("coherence_weight", 0.0),
                laplacian_levels=config["loss"].get("laplacian_levels", 3),
            )
            losses["total_loss"].backward()
            optimizer.step()
            cumulative_loss += losses["total_loss"].item()

        epoch_summary = {
            "epoch": epoch + 1,
            "loss": cumulative_loss / len(training_loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if (epoch + 1) % validation_interval == 0:
            validation_metrics = evaluate_matting_model(
                model,
                validation_loader,
                device,
            )
            epoch_summary["validation"] = validation_metrics
            if validation_metrics["mad"] < best_mad:
                best_mad = validation_metrics["mad"]
                save_joint_checkpoint(
                    model,
                    config["output"]["best_checkpoint"],
                )
            epoch_summary["best_mad"] = best_mad
        print(json.dumps(epoch_summary))

        next_learning_rate = max(
            initial_learning_rate * (0.1 ** (epoch // decay_every)),
            minimum_learning_rate,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = next_learning_rate

    checkpoint_path = save_joint_checkpoint(
        model, config["output"]["final_checkpoint"]
    )
    print(f"saved final joint checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
