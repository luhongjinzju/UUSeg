# -*- coding: utf-8 -*-
"""UCTransNet with two genuinely cascaded CA+SA-enhanced SDI stages.

Data flow:
    encoder -> CCT/Reconstruct -> ASDI stage 1 -> ASDI stage 2
            -> CCA decoder -> segmentation map

Unlike the historical implementation, ASDI stage 2 consumes the four fused
features produced by ASDI stage 1. The two stages do not share parameters.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from CTrans import ChannelTransformer


def get_activation(activation_type: str) -> nn.Module:
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    return nn.ReLU()


class ConvBatchNorm(nn.Module):
    """3x3 convolution followed by batch normalization and activation."""

    def __init__(self, in_channels: int, out_channels: int, activation: str = "ReLU"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


def make_n_conv(
    in_channels: int,
    out_channels: int,
    num_convs: int,
    activation: str = "ReLU",
) -> nn.Sequential:
    layers = [ConvBatchNorm(in_channels, out_channels, activation)]
    for _ in range(num_convs - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class DownBlock(nn.Module):
    """Max-pooling followed by convolution blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int,
        activation: str = "ReLU",
    ):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.convs = make_n_conv(in_channels, out_channels, num_convs, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(self.maxpool(x))


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    """Channel-wise cross attention used by the UCTransNet decoder."""

    def __init__(self, decoder_channels: int, skip_channels: int):
        super().__init__()
        self.mlp_x = nn.Sequential(Flatten(), nn.Linear(skip_channels, skip_channels))
        self.mlp_g = nn.Sequential(Flatten(), nn.Linear(decoder_channels, skip_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        pooled_skip = F.adaptive_avg_pool2d(skip, output_size=1)
        pooled_decoder = F.adaptive_avg_pool2d(decoder, output_size=1)

        skip_descriptor = self.mlp_x(pooled_skip)
        decoder_descriptor = self.mlp_g(pooled_decoder)
        scale = torch.sigmoid((skip_descriptor + decoder_descriptor) / 2.0)
        scale = scale[:, :, None, None]
        return self.relu(skip * scale)


class UpBlockAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int,
        activation: str = "ReLU",
    ):
        super().__init__()
        half_channels = in_channels // 2
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.cca = CCA(decoder_channels=half_channels, skip_channels=half_channels)
        self.convs = make_n_conv(in_channels, out_channels, num_convs, activation)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(x)
        attended_skip = self.cca(decoder=upsampled, skip=skip)
        return self.convs(torch.cat([attended_skip, upsampled], dim=1))


class ChannelAttention(nn.Module):
    """SE-style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mlp(self.avg_pool(x))


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True).values
        scale = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * scale


class ASDIFusion(nn.Module):
    """Fuse four scales into one target scale using CA, SA and Hadamard product.

    Each source feature is processed as follows:
        1x1 channel projection -> channel attention -> spatial attention
        -> scale alignment -> 3x3 smooth convolution.
    The four processed features are fused by element-wise multiplication.
    """

    def __init__(self, source_channels: Sequence[int], target_index: int):
        super().__init__()
        if len(source_channels) != 4:
            raise ValueError("ASDIFusion expects exactly four source scales")

        self.target_index = target_index
        target_channels = source_channels[target_index]

        self.projections = nn.ModuleList(
            [
                nn.Identity()
                if channels == target_channels
                else nn.Conv2d(channels, target_channels, kernel_size=1)
                for channels in source_channels
            ]
        )
        # Independent attention modules allow every source scale to learn its
        # own channel and spatial weighting at this target resolution.
        self.channel_attentions = nn.ModuleList(
            [ChannelAttention(target_channels) for _ in source_channels]
        )
        self.spatial_attentions = nn.ModuleList(
            [SpatialAttention() for _ in source_channels]
        )
        self.smooth_convs = nn.ModuleList(
            [
                nn.Conv2d(target_channels, target_channels, kernel_size=3, padding=1)
                for _ in source_channels
            ]
        )

    @staticmethod
    def _resize_to(x: torch.Tensor, target_size: Sequence[int]) -> torch.Tensor:
        target_h, target_w = int(target_size[0]), int(target_size[1])
        source_h, source_w = x.shape[-2:]

        if source_h > target_h or source_w > target_w:
            return F.adaptive_avg_pool2d(x, output_size=(target_h, target_w))
        if source_h < target_h or source_w < target_w:
            return F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=True,
            )
        return x

    def forward(self, sources: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(sources) != 4:
            raise ValueError(f"Expected four source tensors, got {len(sources)}")

        target_size = sources[self.target_index].shape[-2:]
        fused = None

        for source, projection, ca, sa, smooth in zip(
            sources,
            self.projections,
            self.channel_attentions,
            self.spatial_attentions,
            self.smooth_convs,
        ):
            feature = projection(source)
            feature = ca(feature)
            feature = sa(feature)
            feature = self._resize_to(feature, target_size)
            feature = smooth(feature)
            fused = feature if fused is None else fused * feature

        if fused is None:
            raise RuntimeError("ASDI fusion received no source features")
        return fused


class ASDIStage(nn.Module):
    """One complete four-output ASDI stage."""

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        self.fusions = nn.ModuleList(
            [ASDIFusion(channels, target_index=i) for i in range(4)]
        )

    def forward(self, sources: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [fusion(sources) for fusion in self.fusions]


class UCTransNet(nn.Module):
    """UCTransNet with two independent, serial CA+SA-enhanced ASDI stages."""

    def __init__(
        self,
        config,
        n_channels: int = 1,
        n_classes: int = 1,
        img_size: int = 256,
        vis: bool = False,
        use_sdi: bool = True,
    ):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_sdi = use_sdi

        base = config.base_channel
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        channels = [c1, c2, c3, c4]

        self.inc = ConvBatchNorm(n_channels, c1)
        self.down1 = DownBlock(c1, c2, num_convs=2)
        self.down2 = DownBlock(c2, c3, num_convs=2)
        self.down3 = DownBlock(c3, c4, num_convs=2)
        self.down4 = DownBlock(c4, c4, num_convs=2)

        self.cct = ChannelTransformer(
            config,
            vis,
            img_size,
            channel_num=channels,
            patchSize=config.patch_sizes,
        )

        if use_sdi:
            # The two stages have independent projections, CA, SA and smooth
            # convolutions. Stage 2 consumes the outputs of stage 1.
            self.asdi_stage1 = ASDIStage(channels)
            self.asdi_stage2 = ASDIStage(channels)

        self.up4 = UpBlockAttention(c4 * 2, c3, num_convs=2)
        self.up3 = UpBlockAttention(c3 * 2, c2, num_convs=2)
        self.up2 = UpBlockAttention(c2 * 2, c1, num_convs=2)
        self.up1 = UpBlockAttention(c1 * 2, c1, num_convs=2)

        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        if x.dtype != torch.float32:
            x = x.float()

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1, x2, x3, x4, attention_weights = self.cct(x1, x2, x3, x4)

        if self.use_sdi:
            stage1_features = self.asdi_stage1([x1, x2, x3, x4])
            stage2_features = self.asdi_stage2(stage1_features)
            s1, s2, s3, s4 = stage2_features
        else:
            s1, s2, s3, s4 = x1, x2, x3, x4

        decoded = self.up4(x5, s4)
        decoded = self.up3(decoded, s3)
        decoded = self.up2(decoded, s2)
        decoded = self.up1(decoded, s1)

        logits = self.outc(decoded)
        output = self.last_activation(logits) if self.n_classes == 1 else logits
        return (output, attention_weights) if self.vis else output
