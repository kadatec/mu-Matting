"""Sparse hierarchical matting decoder for four-frame video."""

from __future__ import annotations

import torch
from torch import nn

from ._spconv import replace_feature, spconv
from .sparse_video_resnet import SparseVideoResNet18


class SparseDecoder3D(spconv.SparseModule):
    def __init__(self) -> None:
        super().__init__()
        self.up1 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                512, 256, kernel_size=3, bias=True, indice_key="spconv2"
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
        )
        self.up2 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                320, 256, kernel_size=3, bias=True, indice_key="spconv1"
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
        )
        self.up3 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                320, 64, kernel_size=3, bias=True, indice_key="spconv0"
            ),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
        )
        self.alpha_head = spconv.SparseSequential(
            spconv.SubMConv3d(
                67, 32, kernel_size=3, padding=1, bias=True, indice_key="subm0s"
            ),
            nn.LeakyReLU(inplace=True),
            spconv.SubMConv3d(
                32, 16, kernel_size=3, padding=1, bias=True, indice_key="subm0s"
            ),
            nn.LeakyReLU(inplace=True),
            spconv.SubMConv3d(
                16, 1, kernel_size=1, bias=False, indice_key="subm0s"
            ),
        )
        self.pred_4x = spconv.SubMConv3d(
            256, 1, kernel_size=1, bias=False, indice_key="spconv1"
        )
        self.pred_2x = spconv.SubMConv3d(
            64, 1, kernel_size=1, bias=False, indice_key="spconv0"
        )

    @staticmethod
    def _densify_alpha_logits(sparse_logits):
        activated_logits = replace_feature(
            sparse_logits, torch.sigmoid(sparse_logits.features)
        )
        return activated_logits.dense()

    def forward(self, sparse_input, encoder_features):
        half_resolution_skip, quarter_resolution_skip, _, _, bottleneck = (
            encoder_features
        )

        decoded_4x = self.up1(bottleneck)
        decoded_4x = replace_feature(
            decoded_4x,
            torch.cat(
                (decoded_4x.features, quarter_resolution_skip.features),
                dim=1,
            ),
        )
        decoded_2x = self.up2(decoded_4x)
        quarter_resolution_logits = self.pred_4x(decoded_2x)

        decoded_2x = replace_feature(
            decoded_2x,
            torch.cat(
                (decoded_2x.features, half_resolution_skip.features),
                dim=1,
            ),
        )
        decoded_1x = self.up3(decoded_2x)
        half_resolution_logits = self.pred_2x(decoded_1x)

        normalized_rgb_features = sparse_input.features[:, :3] * 0.5 + 0.5
        decoded_1x = replace_feature(
            decoded_1x,
            torch.cat(
                (decoded_1x.features, normalized_rgb_features), dim=1
            ),
        )
        full_resolution_logits = self.alpha_head(decoded_1x)
        return [
            self._densify_alpha_logits(quarter_resolution_logits),
            self._densify_alpha_logits(half_resolution_logits),
            self._densify_alpha_logits(full_resolution_logits),
        ]


class SHMVideo(nn.Module):
    """Randomly initialized 3D sparse refinement stage."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.backbone = SparseVideoResNet18(in_channels=in_channels)
        self.decoder = SparseDecoder3D()

    def forward(
        self,
        sparse_features,
        sparse_coordinates,
        batch_size: int,
        spatial_shape,
    ):
        sparse_tensor = spconv.SparseConvTensor(
            sparse_features,
            sparse_coordinates.int(),
            spatial_shape,
            batch_size,
        )
        return self.decoder(
            sparse_tensor, self.backbone(sparse_tensor)
        )
