"""Video inference CLI for a final joint checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from .checkpoint import load_joint_checkpoint
from .config import load_config
from .constants import SEQUENCE_LENGTH
from .data import VideoInferenceDataset, save_alpha_png
from .model import build_model
from .runtime import move_batch_to_device, resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Refine one video in four-frame clips")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoint_path = (
        args.checkpoint or config["output"]["final_checkpoint"]
    )
    lowres_size = tuple(config["data"].get("lowres_size", [512, 512]))
    dataset = VideoInferenceDataset(args.video, lowres_size=lowres_size)
    inference_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )
    output_directory = Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    model = build_model(config, load_first_stage_weights=False)
    load_joint_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device).eval()

    for clip_index, batch in enumerate(inference_loader):
        valid_frame_count = int(
            batch.pop("valid_frame_count").item()
        )
        batch = move_batch_to_device(batch, device)
        model_output = model.inference(
            batch["images"], batch["lowres_images"]
        )
        refined_alphas = model_output["refined_alphas"][
            0, 0, :valid_frame_count
        ]
        for local_index, alpha in enumerate(refined_alphas):
            frame_index = clip_index * SEQUENCE_LENGTH + local_index
            save_alpha_png(
                alpha,
                output_directory / f"alpha_{frame_index:06d}.png",
            )
    print(
        f"saved {dataset.frame_count} alpha frames to {output_directory}"
    )


if __name__ == "__main__":
    main()
