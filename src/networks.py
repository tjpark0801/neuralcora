"""Torch implementations of NeuralCORA network architectures and utilities."""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "PeriodicPad2d",
    "PeriodicConv2d",
    "ChannelReLU2d",
    "ChannelSlice",
    "ConvBlock",
    "ResidualBlock",
    "PeriodicConvBlock",
    "NeuralCoraCNN",
    "NeuralCoraResNet",
    "UResNet",
    "UNet",
    "UNetGoogle",
    "build_resnet",
    "build_uresnet",
    "build_unet",
    "build_unet_google",
    "create_lat_mse",
    "create_lat_mae",
    "create_lat_rmse",
    "create_lat_crps",
    "create_lat_crps_lcgev",
    "create_lat_crps_mae",
    "create_lat_log_loss",
    "create_lat_categorical_loss",
    "create_multi_dt_model",
]


def _activation_layer(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name in {"silu", "swish"}:
        return nn.SiLU
    if name in {"leakyrelu", "leaky_relu", "leaky"}:
        return lambda: nn.LeakyReLU(0.1)
    if name == "elu":
        return nn.ELU
    if name == "linear":
        return nn.Identity
    raise ValueError(f"Unsupported activation '{name}'.")


def _flatten_time_dim(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5:
        batch, steps, channels, height, width = x.shape
        return x.reshape(batch, steps * channels, height, width)
    return x


def _validate_sequence(name: str, values: Sequence[int]) -> List[int]:
    if not values:
        raise ValueError(f"{name} must contain at least one element.")
    return list(values)


def _to_device(weights: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return weights.to(device=ref.device, dtype=ref.dtype)


def _lat_weight_tensor(weights: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if ref.ndim < 3:
        raise ValueError("Input tensor must have at least three dimensions.")
    if ref.shape[-2] != weights.numel():
        raise ValueError("Latitude dimension mismatch with provided weights.")
    shape = [1] * ref.ndim
    shape[-2] = weights.numel()
    return _to_device(weights, ref).view(shape)


class PeriodicPad2d(nn.Module):
    """Pad longitude cyclically and latitude with zeros."""

    def __init__(self, pad_width: int) -> None:
        super().__init__()
        self.pad_width = int(pad_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_width == 0:
            return x
        pw = self.pad_width
        left = x[..., -pw:]
        right = x[..., :pw]
        x = torch.cat([left, x, right], dim=-1)
        return F.pad(x, (0, 0, pw, pw))


class PeriodicConv2d(nn.Module):
    """2D convolution with periodic padding along the longitude axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, tuple):
            if kernel_size[0] != kernel_size[1]:
                raise ValueError("PeriodicConv2d only supports square kernels.")
            kernel = kernel_size[0]
        else:
            kernel = kernel_size
        pad_width = (kernel - 1) // 2
        self.pad = PeriodicPad2d(pad_width)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        return self.conv(x)


class ChannelReLU2d(nn.Module):
    """Apply ReLU selectively to a subset of channels."""

    def __init__(self, relu_idxs: Iterable[int]) -> None:
        super().__init__()
        relu_idxs = list(relu_idxs)
        if not relu_idxs:
            raise ValueError("relu_idxs must contain at least one index.")
        self.register_buffer("relu_mask", torch.tensor(sorted(set(relu_idxs)), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        if channels == self.relu_mask.numel():
            return F.relu(x)
        mask = torch.zeros(channels, dtype=torch.bool, device=x.device)
        mask[self.relu_mask] = True
        split = torch.unbind(x, dim=1)
        activated = [
            F.relu(t) if mask[i] else t
            for i, t in enumerate(split)
        ]
        return torch.stack(activated, dim=1)


class ChannelSlice(nn.Module):
    """Slice the first ``n_out`` channels."""

    def __init__(self, n_out: int) -> None:
        super().__init__()
        self.n_out = int(n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, : self.n_out]


class ConvBlock(nn.Module):
    """Conv block with configurable BatchNorm placement."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        bn_position: str | None = None,
        use_bias: bool = True,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        bn_position = bn_position.lower() if bn_position is not None else None
        if bn_position not in {None, "pre", "mid", "post"}:
            raise ValueError("bn_position must be one of None, 'pre', 'mid', or 'post'.")

        self.bn_position = bn_position
        self.bn_pre = nn.BatchNorm2d(in_channels) if bn_position == "pre" else None
        self.conv = PeriodicConv2d(in_channels, out_channels, kernel_size, bias=use_bias)
        self.bn_mid = nn.BatchNorm2d(out_channels) if bn_position == "mid" else None
        self.bn_post = nn.BatchNorm2d(out_channels) if bn_position == "post" else None
        self.activation = _activation_layer(activation)()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bn_pre is not None:
            x = self.bn_pre(x)
        x = self.conv(x)
        if self.bn_mid is not None:
            x = self.bn_mid(x)
        x = self.activation(x)
        if self.bn_post is not None:
            x = self.bn_post(x)
        x = self.dropout(x)
        return x


class PeriodicConvBlock(nn.Module):
    """Simplified periodic convolution block used by the CNN baseline."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.conv = PeriodicConv2d(in_channels, out_channels, kernel_size)
        self.activation = _activation_layer(activation)()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block composed of periodic convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        bn_position: str | None = None,
        use_bias: bool = True,
        dropout: float = 0.0,
        activation: str = "relu",
        skip: bool = True,
        down: bool = False,
        up: bool = False,
    ) -> None:
        super().__init__()
        self.down = down
        self.skip = skip

        self.pool = nn.MaxPool2d(2) if down else nn.Identity()
        self.block1 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            bn_position=bn_position,
            use_bias=use_bias,
            dropout=dropout,
            activation=activation,
        )
        self.block2 = ConvBlock(
            out_channels,
            out_channels,
            kernel_size,
            bn_position=bn_position,
            use_bias=use_bias,
            dropout=dropout,
            activation=activation,
        )

        if skip:
            stride = 2 if down else 1
            self.skip_proj = PeriodicConv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                bias=use_bias,
            )
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        if self.skip_proj is not None:
            residual = self.skip_proj(residual)
        if self.skip:
            x = x + residual
        return x


class NeuralCoraCNN(nn.Module):
    """Simple CNN baseline mirroring the historical TensorFlow setup."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        kernel_size: int,
        out_channels: int,
        *,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_channels = _validate_sequence("hidden_channels", hidden_channels)
        layers: List[nn.Module] = []
        current = in_channels
        for hidden in hidden_channels:
            layers.append(
                PeriodicConvBlock(
                    current,
                    hidden,
                    kernel_size,
                    dropout=dropout,
                    activation=activation,
                )
            )
            current = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = PeriodicConv2d(current, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_time_dim(x)
        features = self.backbone(x)
        return self.head(features)


class NeuralCoraResNet(nn.Module):
    """Residual CNN with periodic padding suitable for NeuralCORA grids."""

    def __init__(
        self,
        input_channels: int,
        filters: Sequence[int],
        kernels: Sequence[int],
        *,
        bn_position: str | None = None,
        use_bias: bool = True,
        dropout: float = 0.0,
        activation: str = "relu",
        skip: bool = True,
        long_skip: bool = False,
        relu_idxs: Sequence[int] | None = None,
        categorical: bool = False,
        nvars: int | None = None,
    ) -> None:
        super().__init__()
        filters = _validate_sequence("filters", filters)
        kernels = _validate_sequence("kernels", kernels)
        if len(filters) != len(kernels):
            raise ValueError("filters and kernels must have equal length.")
        if len(filters) < 2:
            raise ValueError("filters must contain at least two entries.")
        if categorical and nvars is None:
            raise ValueError("nvars must be provided when categorical=True.")

        self.flatten_sequences = True
        self.categorical = categorical
        self.nvars = nvars
        self.relu_layer = ChannelReLU2d(relu_idxs) if relu_idxs is not None else nn.Identity()

        self.input_block = ConvBlock(
            input_channels,
            filters[0],
            kernels[0],
            bn_position=bn_position,
            use_bias=use_bias,
            dropout=dropout,
            activation=activation,
        )

        self.blocks = nn.ModuleList()
        in_channels = filters[0]
        for f, k in zip(filters[1:-1], kernels[1:-1]):
            block = ResidualBlock(
                in_channels,
                f,
                kernel_size=k,
                bn_position=bn_position,
                use_bias=use_bias,
                dropout=dropout,
                activation=activation,
                skip=skip,
            )
            self.blocks.append(block)
            in_channels = f

        self.output_conv = PeriodicConv2d(in_channels, filters[-1], kernels[-1], bias=use_bias)
        self.long_skip = long_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_time_dim(x)
        y = self.input_block(x)
        long_skip = y
        for block in self.blocks:
            y = block(y)
            if self.long_skip:
                y = y + long_skip
        y = self.output_conv(y)
        y = self.relu_layer(y)
        if self.categorical:
            if self.nvars is None:
                raise ValueError("nvars must be set when categorical=True.")
            bins = y.shape[1] // self.nvars
            if bins == 0 or y.shape[1] % self.nvars != 0:
                raise ValueError("Output channels must be divisible by nvars.")
            y = y.view(y.shape[0], self.nvars, bins, *y.shape[2:])
            y = torch.softmax(y, dim=2)
            y = y.view(y.shape[0], self.nvars * bins, *y.shape[3:])
        return y


class UResNet(nn.Module):
    """U-ResNet variant mirroring the legacy TensorFlow implementation."""

    def __init__(
        self,
        input_channels: int,
        filters: Sequence[int],
        kernels: Sequence[int],
        unres: Sequence[int],
        *,
        bn_position: str | None = None,
        use_bias: bool = True,
        dropout: float = 0.0,
        activation: str = "relu",
        skip: bool = True,
    ) -> None:
        super().__init__()
        filters = _validate_sequence("filters", filters)
        kernels = _validate_sequence("kernels", kernels)
        unres = _validate_sequence("unres", unres)

        if len(filters) != len(kernels):
            raise ValueError("filters and kernels must have equal length.")
        if len(unres) == 1:
            unres = [unres[0]] * (len(filters) - 2)
        if len(unres) != len(filters) - 2:
            raise ValueError("unres must have length len(filters) - 2.")

        self.flatten_sequences = True
        self.skip_connections: list[int] = []

        self.stem = ConvBlock(
            input_channels,
            filters[0],
            kernels[0],
            bn_position=bn_position,
            use_bias=use_bias,
            dropout=dropout,
            activation=activation,
        )

        self.stem_blocks = nn.ModuleList()
        in_channels = filters[0]
        for _ in range(unres[0]):
            self.stem_blocks.append(
                ResidualBlock(
                    in_channels,
                    filters[1],
                    kernel_size=kernels[1],
                    bn_position=bn_position,
                    use_bias=use_bias,
                    dropout=dropout,
                    activation=activation,
                    skip=skip,
                )
            )
            in_channels = filters[1]

        self.down_stages = nn.ModuleList()
        current_channels = filters[1]
        self.skip_channels: list[int] = []
        for f, k, nr in zip(filters[2:-1], kernels[2:-1], unres[1:]):
            self.skip_channels.append(current_channels)
            stage = nn.ModuleList()
            for i in range(nr):
                block_in = current_channels if i == 0 else f
                stage.append(
                    ResidualBlock(
                        block_in,
                        f,
                        kernel_size=k,
                        bn_position=bn_position,
                        use_bias=use_bias,
                        dropout=dropout,
                        activation=activation,
                        skip=skip,
                        down=i == 0,
                    )
                )
                current_channels = f
            self.down_stages.append(stage)

        self.up_stages = nn.ModuleList()
        for skip_channels, f, k, nr in zip(
            reversed(self.skip_channels),
            reversed(filters[1:-2]),
            reversed(kernels[1:-2]),
            reversed(unres[:-1]),
        ):
            stage = nn.ModuleList()
            for i in range(nr):
                block_in = current_channels + skip_channels if i == 0 else f
                stage.append(
                    ResidualBlock(
                        block_in,
                        f,
                        kernel_size=k,
                        bn_position=bn_position,
                        use_bias=use_bias,
                        dropout=dropout,
                        activation=activation,
                        skip=skip,
                        up=i == 0,
                    )
                )
                current_channels = f
            self.up_stages.append(stage)

        self.output_conv = PeriodicConv2d(current_channels, filters[-1], kernels[-1], bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_time_dim(x)
        y = self.stem(x)
        for block in self.stem_blocks:
            y = block(y)

        skips: list[torch.Tensor] = []
        for stage in self.down_stages:
            skips.append(y)
            for block in stage:
                y = block(y)

        for skip, stage in zip(reversed(skips), self.up_stages):
            y = F.interpolate(y, scale_factor=2, mode="nearest")
            y = torch.cat([skip, y], dim=1)
            for block in stage:
                y = block(y)

        return self.output_conv(y)


class UNet(nn.Module):
    """UNet with residual blocks and periodic convolutions."""

    def __init__(
        self,
        input_channels: int,
        n_layers: int,
        filters_start: int,
        channels_out: int,
        *,
        kernel: int = 3,
        u_skip: bool = True,
        res_skip: bool = True,
        bn_position: str | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")

        self.flatten_sequences = True
        self.u_skip = u_skip
        self.res_skip = res_skip
        self.relu = nn.ReLU()

        filters_per_layer = [filters_start * (2**i) for i in range(n_layers)]

        self.down = nn.ModuleList()
        in_channels = input_channels
        for i, filters in enumerate(filters_per_layer):
            block = nn.ModuleDict(
                {
                    "residual": PeriodicConv2d(in_channels, filters, 1, bias=False),
                    "block1": ConvBlock(
                        in_channels,
                        filters,
                        kernel,
                        bn_position=bn_position,
                        dropout=dropout,
                        activation=activation,
                    ),
                    "block2": ConvBlock(
                        filters,
                        filters,
                        kernel,
                        bn_position=bn_position,
                        dropout=dropout,
                        activation="linear",
                    ),
                    "pool": nn.MaxPool2d(2) if i != n_layers - 1 else nn.Identity(),
                }
            )
            self.down.append(block)
            in_channels = filters

        self.up = nn.ModuleList()
        current_filters = filters_per_layer[-1]
        for filters in reversed(filters_per_layer[:-1]):
            concat_channels = current_filters + filters if u_skip else current_filters
            stage = nn.ModuleDict(
                {
                    "upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                    "conv": PeriodicConv2d(current_filters, filters, 3),
                    "residual": PeriodicConv2d(
                        concat_channels if u_skip else filters,
                        filters,
                        1,
                        bias=False,
                    ),
                    "block1": ConvBlock(
                        concat_channels if u_skip else filters,
                        filters,
                        kernel,
                        bn_position=bn_position,
                        dropout=dropout,
                        activation=activation,
                    ),
                    "block2": ConvBlock(
                        filters,
                        filters,
                        kernel,
                        bn_position=bn_position,
                        dropout=dropout,
                        activation="linear",
                    ),
                }
            )
            self.up.append(stage)
            current_filters = filters

        self.head = PeriodicConv2d(filters_per_layer[0], channels_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_time_dim(x)
        skips: list[torch.Tensor] = []

        for stage in self.down:
            residual = stage["residual"](x)
            y = stage["block1"](x)
            y = stage["block2"](y)
            if self.res_skip:
                y = y + residual
            y = self.relu(y)
            skips.append(y)
            x = stage["pool"](y)

        x = skips.pop()  # bottleneck output
        for stage in self.up:
            x = stage["upsample"](x)
            x = stage["conv"](x)
            x = self.relu(x)
            if self.u_skip:
                skip = skips.pop()
                x = torch.cat([skip, x], dim=1)
            residual = stage["residual"](x)
            y = stage["block1"](x)
            y = stage["block2"](y)
            if self.res_skip:
                y = y + residual
            x = self.relu(y)

        return self.head(x)


class _GoogleBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = PeriodicConv2d(in_channels, out_channels, 3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU()
        self.conv2 = PeriodicConv2d(out_channels, out_channels, 3)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = PeriodicConv2d(in_channels, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.drop(y)
        return y + residual


class _GoogleDownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act2 = nn.LeakyReLU()
        self.conv = PeriodicConv2d(in_channels, out_channels, 3)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = PeriodicConv2d(in_channels, out_channels, 3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        y = self.bn1(x)
        y = self.act1(y)
        y = self.pool(y)
        y = self.bn2(y)
        y = self.act2(y)
        y = self.conv(y)
        y = self.drop(y)
        return y + residual


class _GoogleUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        total_channels = in_channels + skip_channels
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.bn1 = nn.BatchNorm2d(total_channels)
        self.act1 = nn.LeakyReLU()
        self.conv1 = PeriodicConv2d(total_channels, out_channels, 3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU()
        self.conv2 = PeriodicConv2d(out_channels, out_channels, 3)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = PeriodicConv2d(total_channels, out_channels, 3)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        y = torch.cat([x, skip], dim=1)
        y = self.upsample(y)
        residual = self.shortcut(y)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.act2(y)
        y = self.conv2(y)
        y = self.drop(y)
        return y + residual


class UNetGoogle(nn.Module):
    """UNet variant inspired by Agrawal et al."""

    def __init__(
        self,
        input_channels: int,
        filters: Sequence[int],
        output_channels: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        filters = _validate_sequence("filters", filters)
        if len(filters) < 2:
            raise ValueError("filters must contain at least two entries.")

        self.flatten_sequences = True

        self.initial = _GoogleBasicBlock(input_channels, filters[0], dropout)

        self.encoder = nn.ModuleList()
        current_channels = filters[0]
        for f in filters[:-1]:
            self.encoder.append(_GoogleDownsampleBlock(current_channels, f, dropout))
            current_channels = f

        self.bottleneck = _GoogleBasicBlock(current_channels, filters[-1], dropout)
        current_channels = filters[-1]

        self.decoder = nn.ModuleList()
        for f in reversed(filters[:-1]):
            self.decoder.append(_GoogleUpsampleBlock(current_channels, f, f, dropout))
            current_channels = f

        self.output_conv = PeriodicConv2d(current_channels, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_time_dim(x)
        skips: list[torch.Tensor] = []
        x = self.initial(x)
        for block in self.encoder:
            skips.append(x)
            x = block(x)

        x = self.bottleneck(x)

        for block, skip in zip(self.decoder, reversed(skips)):
            x = block(x, skip)

        return self.output_conv(x)


def build_resnet(
    filters: Sequence[int],
    kernels: Sequence[int],
    input_shape: Sequence[int],
    *,
    bn_position: str | None = None,
    use_bias: bool = True,
    skip: bool = True,
    dropout: float = 0.0,
    activation: str = "relu",
    long_skip: bool = False,
    relu_idxs: Sequence[int] | None = None,
    categorical: bool = False,
    nvars: int | None = None,
) -> NeuralCoraResNet:
    channels = input_shape[-1]
    return NeuralCoraResNet(
        channels,
        filters,
        kernels,
        bn_position=bn_position,
        use_bias=use_bias,
        dropout=dropout,
        activation=activation,
        skip=skip,
        long_skip=long_skip,
        relu_idxs=relu_idxs,
        categorical=categorical,
        nvars=nvars,
    )


def build_uresnet(
    filters: Sequence[int],
    kernels: Sequence[int],
    unres: Sequence[int],
    input_shape: Sequence[int],
    *,
    bn_position: str | None = None,
    use_bias: bool = True,
    skip: bool = True,
    dropout: float = 0.0,
    activation: str = "relu",
) -> UResNet:
    channels = input_shape[-1]
    return UResNet(
        channels,
        filters,
        kernels,
        unres,
        bn_position=bn_position,
        use_bias=use_bias,
        dropout=dropout,
        activation=activation,
        skip=skip,
    )


def build_unet(
    input_shape: Sequence[int],
    n_layers: int,
    filters_start: int,
    channels_out: int,
    *,
    kernel: int = 3,
    u_skip: bool = True,
    res_skip: bool = True,
    bn_position: str | None = None,
    dropout: float = 0.0,
    activation: str = "relu",
) -> UNet:
    channels = input_shape[-1]
    return UNet(
        channels,
        n_layers,
        filters_start,
        channels_out,
        kernel=kernel,
        u_skip=u_skip,
        res_skip=res_skip,
        bn_position=bn_position,
        dropout=dropout,
        activation=activation,
    )


def build_unet_google(
    filters: Sequence[int],
    input_shape: Sequence[int],
    output_channels: int,
    *,
    dropout: float = 0.0,
) -> UNetGoogle:
    channels = input_shape[-1]
    return UNetGoogle(
        channels,
        filters,
        output_channels,
        dropout=dropout,
    )


def create_lat_mse(lat) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    weights_lat = np.cos(np.deg2rad(lat)).astype("float32")
    weights_lat /= weights_lat.mean()
    weights = torch.tensor(weights_lat, dtype=torch.float32)

    def lat_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        w = _lat_weight_tensor(weights, y_true)
        error = y_true - y_pred
        mse = error.pow(2) * w
        return mse.mean()

    return lat_mse


def create_lat_mae(lat) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    weights_lat = np.cos(np.deg2rad(lat)).astype("float32")
    weights_lat /= weights_lat.mean()
    weights = torch.tensor(weights_lat, dtype=torch.float32)

    def lat_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        w = _lat_weight_tensor(weights, y_true)
        error = (y_true - y_pred).abs()
        mae = error * w
        return mae.mean()

    return lat_mae


def create_lat_rmse(lat) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    weights_lat = np.cos(np.deg2rad(lat)).astype("float32")
    weights_lat /= weights_lat.mean()
    weights = torch.tensor(weights_lat, dtype=torch.float32)

    def lat_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        w = _lat_weight_tensor(weights, y_true)
        error = y_true - y_pred
        mse = (error.pow(2) * w).mean(dim=tuple(range(1, y_true.ndim)))
        return torch.sqrt(mse).mean()

    return lat_rmse


def create_lat_crps(lat, n_vars: int, relu: bool = False) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    weights_lat = np.cos(np.deg2rad(lat)).astype("float32")
    weights_lat /= weights_lat.mean()
    weights = torch.tensor(weights_lat, dtype=torch.float32)

    def crps_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        w = _lat_weight_tensor(weights, y_true)
        mu, sigma = y_pred[:, :n_vars], y_pred[:, n_vars:]
        sigma = torch.relu(sigma) if relu else torch.sqrt(torch.clamp(sigma.pow(2), min=1e-7))
        loc = (y_true - mu) / torch.clamp(sigma, min=1e-7)
        phi = (1.0 / np.sqrt(2.0 * np.pi)) * torch.exp(-0.5 * loc.pow(2))
        Phi = 0.5 * (1.0 + torch.erf(loc / np.sqrt(2.0)))
        crps = sigma * (loc * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
        crps = crps * w
        return crps.mean()

    return crps_loss


def gev_cdf_torch(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    y = (y - mu) / sigma
    x = 1 + xi * y
    x = torch.where(x < 0, torch.zeros_like(x), x)
    x = torch.pow(x, -1 / xi)
    result = torch.exp(-x)
    result = torch.where(torch.isinf(result), torch.zeros_like(result), result)
    return result


def crps_lcgev_torch(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    sigma = sigma / xi
    gam = torch.exp(torch.lgamma(1 - xi))

    prob_y = gev_cdf_torch(y, mu, sigma, xi)
    prob_0 = gev_cdf_torch(torch.zeros_like(y), mu, sigma, xi)

    igamma_y = torch.special.gammainc(1 - xi.double(), (-torch.log(prob_y.double()))).to(y.dtype)
    igamma_0 = torch.special.gammainc(1 - xi.double(), (-2 * torch.log(prob_0.double()))).to(y.dtype)

    term1 = (y - mu) * (2 * prob_y - 1) + mu * prob_0.pow(2)
    term2 = sigma * (1 - prob_0.pow(2) - (2**xi) * gam * igamma_0)
    term3 = -2 * sigma * (1 - prob_y - gam * igamma_y)
    return term1 + term2 + term3


def create_lat_crps_lcgev(lat, n_vars: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    weights_lat = np.cos(np.deg2rad(lat)).astype("float32")
    weights_lat /= weights_lat.mean()
    weights = torch.tensor(weights_lat, dtype=torch.float32)

    def crps_lcgev_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        w = _lat_weight_tensor(weights, y_true).squeeze(1)
        mu = y_pred[:, 0]
        sigma = torch.relu(y_pred[:, 1])
        xi = torch.clamp(y_pred[:, 2], -0.278, 0.999)
        target = y_true[:, 0] if y_true.ndim > 3 else y_true
        loss = crps_lcgev_torch(target, mu, sigma, xi) * w
        return loss.mean()

    return crps_lcgev_loss


def create_lat_crps_mae(lat, n_vars: int, beta: float = 1.0) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    weights_lat = np.cos(np.deg2rad(lat)).astype("float32")
    weights_lat /= weights_lat.mean()
    weights = torch.tensor(weights_lat, dtype=torch.float32)

    def crps_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        w = _lat_weight_tensor(weights, y_true)
        mu = y_pred[:, :n_vars]
        sigma = torch.relu(y_pred[:, n_vars:])
        loc = (y_true - mu) / torch.clamp(sigma, min=1e-7)
        phi = (1.0 / np.sqrt(2.0 * np.pi)) * torch.exp(-0.5 * loc.pow(2))
        Phi = 0.5 * (1.0 + torch.erf(loc / np.sqrt(2.0)))
        crps = sigma * (loc * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
        crps = crps * w
        crps = crps.mean()

        mae = (y_true - mu).abs() * w
        mae = mae.mean()

        return crps + beta * mae

    return crps_mae


def create_lat_log_loss(lat, n_vars: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    weights_lat = np.cos(np.deg2rad(lat)).astype("float32")
    weights_lat /= weights_lat.mean()
    weights = torch.tensor(weights_lat, dtype=torch.float32)

    def log_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        w = _lat_weight_tensor(weights, y_true)
        mu = y_pred[:, :n_vars]
        sigma = torch.relu(y_pred[:, n_vars:])
        eps = 1e-7
        sigma = torch.clamp(sigma, min=eps)
        prob = (1 / (sigma * np.sqrt(2 * np.pi))) * torch.exp(-0.5 * ((y_true - mu) / sigma) ** 2)
        ll = -torch.log(torch.clamp(prob, min=eps))
        ll = ll * w
        return ll.mean()

    return log_loss


def create_lat_categorical_loss(lat, n_vars: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    weights_lat = np.cos(np.deg2rad(lat)).astype("float32")
    weights_lat /= weights_lat.mean()
    weights = torch.tensor(weights_lat, dtype=torch.float32)

    def categorical_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        w = _lat_weight_tensor(weights, y_true)
        batch, channels, height, width = y_pred.shape
        bins = channels // n_vars
        if bins == 0 or channels % n_vars != 0:
            raise ValueError("y_pred channels must be divisible by n_vars.")
        probs = y_pred.view(batch, n_vars, bins, height, width)
        targets = y_true.view(batch, n_vars, bins, height, width)
        log_probs = torch.log(torch.clamp(probs, min=1e-7))
        loss = -(targets * log_probs).sum(dim=2) * w
        return loss.mean()

    return categorical_loss


class MultiStepModel(nn.Module):
    """Wrap a base model to produce multi-step predictions."""

    def __init__(self, base_model: nn.Module, multi_dt: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.multi_dt = int(multi_dt)

    def forward(self, inputs: torch.Tensor, const_inputs: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        x = inputs
        for _ in range(self.multi_dt):
            combined = torch.cat([x, const_inputs], dim=1)
            x = self.base_model(combined)
            outputs.append(x)
        return outputs


def create_multi_dt_model(model: nn.Module, multi_dt: int) -> MultiStepModel:
    return MultiStepModel(model, multi_dt)
