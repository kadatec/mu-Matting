"""FAR evaluation CLI."""

from __future__ import annotations

import argparse
import json

from .checkpoint import load_joint_checkpoint
from .config import load_config
from .model import build_model
from .runtime import (
    create_matting_data_loader,
    evaluate_matting_model,
    resolve_device,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a FAR checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoint_path = (
        args.checkpoint or config["output"]["final_checkpoint"]
    )
    evaluation_loader = create_matting_data_loader(
        config, "evaluation", shuffle=False
    )
    model = build_model(config, load_first_stage_weights=False)
    load_joint_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    metrics = evaluate_matting_model(model, evaluation_loader, device)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
