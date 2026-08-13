"""Two-stage sparse video alpha model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .constants import SEQUENCE_LENGTH
from .shm_video import SHMVideo


class FrozenMMSegAlphaPredictor(nn.Module):
    """Frozen MMEngine/MMSeg alpha predictor loaded exactly once."""

    def __init__(
        self, config_path: str, checkpoint_path: str | None
    ) -> None:
        super().__init__()
        try:
            from mmseg.apis import init_model
        except ImportError as exc:
            raise ImportError("the configured first stage requires mmsegmentation") from exc
        self.model = init_model(config_path, checkpoint_path)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        from mmseg.structures import SegDataSample

        height, width = images.shape[-2:]
        input_samples = []
        for _ in range(images.shape[0]):
            sample = SegDataSample()
            sample.set_metainfo(
                {
                    "ori_shape": (height, width),
                    "img_shape": (height, width),
                    "pad_shape": (height, width),
                }
            )
            input_samples.append(sample)
        output_samples = self.model.test_step(
            {
                "inputs": [image for image in images],
                "data_samples": input_samples,
            }
        )
        predictions = []
        for data_sample in output_samples:
            if hasattr(data_sample, "pred_depth_map"):
                prediction = data_sample.pred_depth_map.data
            elif hasattr(data_sample, "pred_sem_seg"):
                prediction = data_sample.pred_sem_seg.data
            else:
                raise TypeError("CAP result has no supported prediction field")
            if prediction.ndim == 2:
                prediction = prediction.unsqueeze(0)
            predictions.append(prediction.float())
        return torch.stack(predictions)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self


class SparseVideoAlphaRefiner(nn.Module):
    """Frozen first stage plus randomly initialized sparse 3D second stage."""

    sequence_length = SEQUENCE_LENGTH

    def __init__(
        self,
        first_stage: nn.Module,
        dilation_kernel: int = 5,
        refiner: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if dilation_kernel < 1 or dilation_kernel % 2 == 0:
            raise ValueError("dilation_kernel must be a positive odd integer")
        self.first_stage = first_stage
        self.refiner = SHMVideo(in_channels=4) if refiner is None else refiner
        self.dilate_pool = nn.MaxPool2d(
            dilation_kernel, stride=1, padding=dilation_kernel // 2
        )
        for parameter in self.first_stage.parameters():
            parameter.requires_grad_(False)
        self.first_stage.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.first_stage.eval()
        return self

    def _validate_inputs(
        self,
        images: torch.Tensor,
        lowres_images: torch.Tensor,
        crop_boxes: torch.Tensor | None,
    ) -> None:
        if images.ndim != 5 or images.shape[1] != 3:
            raise ValueError("images must have shape B,3,T,H,W")
        if lowres_images.ndim != 5 or lowres_images.shape[2] != 3:
            raise ValueError("lowres_images must have shape B,T,3,h,w")
        if images.shape[2] != self.sequence_length:
            raise ValueError(
                "images must contain exactly "
                f"{self.sequence_length} frames"
            )
        if lowres_images.shape[:2] != (images.shape[0], self.sequence_length):
            raise ValueError(
                "images and lowres_images must have matching batch and "
                "temporal dimensions"
            )
        if crop_boxes is not None:
            self._validate_crop_boxes(
                crop_boxes, images.shape[0], self.sequence_length
            )

    @staticmethod
    def _validate_crop_boxes(
        crop_boxes: torch.Tensor,
        batch_size: int,
        frame_count: int,
    ) -> None:
        if not isinstance(crop_boxes, torch.Tensor):
            raise TypeError("crop_boxes must be a tensor")
        if crop_boxes.shape != (batch_size, frame_count, 4):
            raise ValueError("crop_boxes must have shape B,T,4")
        if not torch.is_floating_point(crop_boxes):
            raise TypeError("crop_boxes must use a floating-point dtype")
        if not torch.isfinite(crop_boxes).all():
            raise ValueError("crop_boxes must contain only finite values")
        if ((crop_boxes < 0) | (crop_boxes > 1)).any():
            raise ValueError("crop_boxes coordinates must lie in [0, 1]")
        left, top, right, bottom = crop_boxes.unbind(dim=-1)
        if ((left >= right) | (top >= bottom)).any():
            raise ValueError(
                "crop_boxes must satisfy x1 < x2 and y1 < y2"
            )

    @torch.no_grad()
    def _predict_coarse_alphas(
        self, lowres_images: torch.Tensor
    ) -> torch.Tensor:
        batch_size, frame_count, channels, height, width = lowres_images.shape
        flattened_images = lowres_images.reshape(
            batch_size * frame_count, channels, height, width
        )
        coarse_alphas = self.first_stage(flattened_images)
        if coarse_alphas.ndim == 5 and coarse_alphas.shape[1] == 1:
            coarse_alphas = coarse_alphas.squeeze(1)
        if coarse_alphas.ndim != 4 or coarse_alphas.shape[1] != 1:
            raise ValueError("first stage must return B*T,1,h,w alpha predictions")
        return coarse_alphas.reshape(
            batch_size,
            frame_count,
            1,
            *coarse_alphas.shape[-2:],
        ).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _resize_coarse_alphas(
        coarse_alphas: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, _, frame_count, height, width = coarse_alphas.shape
        flattened_alphas = coarse_alphas.permute(
            0, 2, 1, 3, 4
        ).reshape(batch_size * frame_count, 1, height, width)
        resized_alphas = F.interpolate(
            flattened_alphas,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return resized_alphas.reshape(
            batch_size, frame_count, 1, *output_size
        ).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _crop_and_resize_coarse_alphas(
        coarse_alphas: torch.Tensor,
        crop_boxes: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, _, frame_count, height, width = coarse_alphas.shape
        flattened_alphas = coarse_alphas.permute(
            0, 2, 1, 3, 4
        ).reshape(batch_size * frame_count, 1, height, width)
        coordinate_scale = crop_boxes.new_tensor(
            (width, height, width, height)
        )
        pixel_boxes = (
            crop_boxes.reshape(-1, 4) * coordinate_scale
        ).to(torch.int64)
        left, top, right, bottom = pixel_boxes.unbind(dim=-1)
        empty_slices = (left >= right) | (top >= bottom)
        if empty_slices.any():
            invalid_indices = (
                empty_slices.nonzero().flatten().detach().cpu().tolist()
            )
            raise ValueError(
                "crop_boxes produce empty coarse-alpha slices at flattened "
                f"indices {invalid_indices}"
            )

        cropped_alphas = []
        for index, (left, top, right, bottom) in enumerate(
            pixel_boxes.detach().cpu().tolist()
        ):
            alpha_patch = flattened_alphas[
                index : index + 1, :, top:bottom, left:right
            ]
            cropped_alphas.append(
                F.interpolate(
                    alpha_patch,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.cat(cropped_alphas, dim=0).reshape(
            batch_size, frame_count, 1, *output_size
        ).permute(0, 2, 1, 3, 4)

    @classmethod
    def _align_coarse_alphas(
        cls,
        coarse_alphas: torch.Tensor,
        crop_boxes: torch.Tensor | None,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if crop_boxes is None:
            return cls._resize_coarse_alphas(
                coarse_alphas, output_size
            )
        return cls._crop_and_resize_coarse_alphas(
            coarse_alphas, crop_boxes, output_size
        )

    def _create_unknown_region_mask(
        self, coarse_alphas: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, frame_count, height, width = coarse_alphas.shape
        flattened_alphas = coarse_alphas.permute(
            0, 2, 1, 3, 4
        ).reshape(batch_size * frame_count, 1, height, width)
        unknown_regions = (
            (flattened_alphas > 0.01) & (flattened_alphas < 0.99)
        ).to(coarse_alphas.dtype)
        return self.dilate_pool(unknown_regions).reshape(
            batch_size, frame_count, 1, height, width
        ).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _extract_sparse_features(
        images: torch.Tensor,
        coarse_alphas: torch.Tensor,
        unknown_region_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dense_features = torch.cat((images, coarse_alphas), dim=1)
        active_indices = torch.where(unknown_region_mask.squeeze(1) > 0)
        sparse_features = dense_features.permute(0, 2, 3, 4, 1)[
            active_indices
        ]
        sparse_coordinates = torch.stack(active_indices, dim=1)
        return sparse_features, sparse_coordinates

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        images = batch["images"]
        lowres_images = batch["lowres_images"]
        crop_boxes = batch.get("crop_boxes")
        self._validate_inputs(images, lowres_images, crop_boxes)
        full_frame_coarse_alphas = self._predict_coarse_alphas(
            lowres_images
        ).to(images.device)
        coarse_alphas = self._align_coarse_alphas(
            full_frame_coarse_alphas,
            crop_boxes,
            images.shape[-2:],
        )
        unknown_region_mask = self._create_unknown_region_mask(coarse_alphas)
        sparse_features, sparse_coordinates = self._extract_sparse_features(
            images, coarse_alphas, unknown_region_mask
        )
        multiscale_alphas = self.refiner(
            sparse_features,
            sparse_coordinates,
            images.shape[0],
            unknown_region_mask.shape[2:],
        )
        refined_alphas = (
            multiscale_alphas[-1] * unknown_region_mask
            + coarse_alphas * (1 - unknown_region_mask)
        )
        return {
            "multiscale_alphas": multiscale_alphas,
            "refined_alphas": refined_alphas,
            "coarse_alphas": coarse_alphas,
            "unknown_region_mask": unknown_region_mask,
        }

    @torch.no_grad()
    def inference(self, images: torch.Tensor, lowres_images: torch.Tensor):
        was_training = self.training
        self.eval()
        try:
            return self(
                {"images": images, "lowres_images": lowres_images}
            )
        finally:
            self.train(was_training)


def build_model(
    config: dict, *, load_first_stage_weights: bool
) -> SparseVideoAlphaRefiner:
    model_config = config["model"]
    first_stage = FrozenMMSegAlphaPredictor(
        model_config["first_stage_config"],
        model_config["first_stage_checkpoint"]
        if load_first_stage_weights
        else None,
    )
    return SparseVideoAlphaRefiner(
        first_stage, dilation_kernel=model_config.get("dilation_kernel", 5)
    )
