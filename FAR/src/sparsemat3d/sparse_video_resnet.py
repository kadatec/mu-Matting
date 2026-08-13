"""Sparse 3D ResNet backbone used by the four-frame refiner."""

from __future__ import annotations

from torch import nn

from ._spconv import replace_feature, spconv


class BasicBlock3D(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample=None,
        dilation: int = 1,
        padding: int = 1,
        indice_key: str | None = None,
        last_indice_key: str | None = None,
    ) -> None:
        super().__init__()
        convolution_class = (
            spconv.SparseConv3d
            if stride == 2
            else spconv.SubMConv3d
        )
        self.conv1 = convolution_class(
            in_channels,
            out_channels,
            3,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=False,
            indice_key=indice_key,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=True,
            indice_key=last_indice_key,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, sparse_input):
        residual = (
            sparse_input
            if self.downsample is None
            else self.downsample(sparse_input)
        )
        output = self.conv1(sparse_input)
        output = replace_feature(
            output, self.relu(self.bn1(output.features))
        )
        output = self.conv2(output)
        output = replace_feature(output, self.bn2(output.features))
        return replace_feature(
            output,
            self.relu(output.features + residual.features),
        )


class SparseVideoResNet18(nn.Module):
    """SparseMat-style 3D topology with spatial output stride eight.

    The first two residual stages downsample all three dimensions. For a
    four-frame input, temporal sizes therefore follow 4 -> 2 -> 1.
    """

    def __init__(
        self,
        in_channels: int = 4,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = spconv.SubMConv3d(
            in_channels,
            64,
            kernel_size=3,
            padding=1,
            bias=False,
            indice_key="subm0s",
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spconv.SparseConv3d(
            64,
            64,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
            bias=False,
            indice_key="spconv0",
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = spconv.SubMConv3d(
            64, 64, kernel_size=3, padding=1, bias=False, indice_key="subm0e"
        )
        self.bn3 = nn.BatchNorm1d(64)

        self.layer1 = self._build_stage(64, layers[0], 2, "spconv1")
        self.layer2 = self._build_stage(128, layers[1], 2, "spconv2")
        self.layer3 = self._build_stage(256, layers[2], 1, "spconv3")
        self.layer4 = self._build_stage(
            512, layers[3], 1, "spconv4", dilation=2, padding=2
        )

        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_stage(
        self,
        out_channels: int,
        block_count: int,
        stride: int,
        indice_key: str,
        dilation: int = 1,
        padding: int = 1,
    ):
        convolution_class = (
            spconv.SparseConv3d
            if stride == 2
            else spconv.SubMConv3d
        )
        downsample = spconv.SparseSequential(
            convolution_class(
                self.in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            nn.BatchNorm1d(out_channels),
        )
        stage_blocks = [
            BasicBlock3D(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                dilation=dilation,
                padding=padding,
                indice_key=indice_key,
                last_indice_key=f"subm{indice_key[-1]}e",
            )
        ]
        self.in_channels = out_channels
        stage_blocks.extend(
            BasicBlock3D(out_channels, out_channels)
            for _ in range(1, block_count)
        )
        return spconv.SparseSequential(*stage_blocks)

    def forward(self, sparse_input):
        output = self.conv1(sparse_input)
        output = replace_feature(
            output, self.relu(self.bn1(output.features))
        )
        output = self.conv2(output)
        output = replace_feature(
            output, self.relu(self.bn2(output.features))
        )
        output = self.conv3(output)
        output = replace_feature(
            output, self.relu(self.bn3(output.features))
        )

        stage_outputs = [output]
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            output = layer(output)
            stage_outputs.append(output)
        return stage_outputs
