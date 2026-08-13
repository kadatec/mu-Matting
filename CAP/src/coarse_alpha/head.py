from typing import Optional, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList
from torch import Tensor

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class ViTCoarseAlphaHead(BaseDecodeHead):
    """Sapiens ViT upsampling head producing a coarse alpha matte."""

    def __init__(
        self,
        deconv_out_channels: OptIntSeq = (256, 256, 256),
        deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
        conv_out_channels: OptIntSeq = None,
        conv_kernel_sizes: OptIntSeq = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.deconv_layers = self._make_deconv_layers(
            self.in_channels,
            deconv_out_channels,
            deconv_kernel_sizes,
        )
        current_channels = (
            deconv_out_channels[-1]
            if deconv_out_channels else self.in_channels
        )
        self.conv_layers = self._make_conv_layers(
            current_channels,
            conv_out_channels,
            conv_kernel_sizes,
        )

    @staticmethod
    def _validate_layers(
        channels: OptIntSeq,
        kernels: OptIntSeq,
        layer_name: str,
    ) -> None:
        if (channels is None) != (kernels is None):
            raise ValueError(
                f'{layer_name} channels and kernels must both be provided')
        if channels is None:
            return
        if len(channels) != len(kernels):
            raise ValueError(
                f'{layer_name} channels and kernels must have equal lengths')
        if any(value < 1 for value in channels):
            raise ValueError(
                f'{layer_name} channels must contain positive integers')
        if any(value < 1 for value in kernels):
            raise ValueError(
                f'{layer_name} kernels must contain positive integers')

    def _make_deconv_layers(
        self,
        in_channels: int,
        out_channels: OptIntSeq,
        kernel_sizes: OptIntSeq,
    ) -> nn.Module:
        self._validate_layers(out_channels, kernel_sizes, 'deconvolution')
        if not out_channels:
            return nn.Identity()

        layers = []
        for channels, kernel_size in zip(out_channels, kernel_sizes):
            if kernel_size == 4:
                padding, output_padding = 1, 0
            elif kernel_size == 3:
                padding, output_padding = 1, 1
            elif kernel_size == 2:
                padding, output_padding = 0, 0
            else:
                raise ValueError(
                    f'Unsupported deconvolution kernel size {kernel_size}')
            layers.extend([
                build_upsample_layer(dict(
                    type='deconv',
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )),
                nn.InstanceNorm2d(channels),
                nn.SiLU(inplace=True),
            ])
            in_channels = channels
        return nn.Sequential(*layers)

    def _make_conv_layers(
        self,
        in_channels: int,
        out_channels: OptIntSeq,
        kernel_sizes: OptIntSeq,
    ) -> nn.Module:
        self._validate_layers(out_channels, kernel_sizes, 'convolution')
        if not out_channels:
            return nn.Identity()

        layers = []
        for channels, kernel_size in zip(out_channels, kernel_sizes):
            layers.extend([
                build_conv_layer(dict(
                    type='Conv2d',
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                )),
                nn.InstanceNorm2d(channels),
                nn.SiLU(inplace=True),
            ])
            in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, inputs) -> Tensor:
        features = self._transform_inputs(inputs)
        features = self.deconv_layers(features)
        features = self.conv_layers(features)
        return torch.sigmoid(self.cls_seg(features))

    @staticmethod
    def _stack_batch_gt(batch_data_samples: SampleList) -> Tensor:
        return torch.stack([
            data_sample.gt_depth_map.data
            for data_sample in batch_data_samples
        ])

    def loss(
        self,
        inputs,
        batch_data_samples: SampleList,
        train_cfg: ConfigType,
    ) -> dict:
        return self.loss_by_feat(
            self.forward(inputs), batch_data_samples)

    def loss_by_feat(
        self,
        predictions: Tensor,
        batch_data_samples: SampleList,
    ) -> dict:
        targets = self._stack_batch_gt(batch_data_samples)
        predictions = resize(
            input=predictions,
            size=targets.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
        )
        pixel_weight = (
            self.sampler.sample(predictions, targets)
            if self.sampler is not None else None
        )
        losses = {}
        loss_modules = (
            self.loss_decode
            if isinstance(self.loss_decode, nn.ModuleList)
            else [self.loss_decode]
        )
        for loss_module in loss_modules:
            value = loss_module(
                predictions, targets, weight=pixel_weight)
            losses[loss_module.loss_name] = (
                losses.get(loss_module.loss_name, 0) + value
            )
        return losses
