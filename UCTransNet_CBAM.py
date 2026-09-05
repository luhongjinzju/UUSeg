# -*- coding: utf-8 -*-
"""UCTransNet + full CBAM on the four reconstructed skip features.

Data flow:
    encoder -> CCT/reconstruction -> CBAM -> CCA decoder -> segmentation map

This file intentionally contains no SDI module.  ``use_sdi`` is retained only
for compatibility with the existing ``build_uctransnet`` wrapper, which passes
that keyword when constructing every UCTransNet variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from CTrans import ChannelTransformer


def get_activation(activation_type: str):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    return nn.ReLU()


class ConvBatchNorm(nn.Module):
    """3x3 convolution followed by batch normalization and activation."""

    def __init__(self, in_channels, out_channels, activation="ReLU"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


def _make_nConv(in_channels, out_channels, nb_Conv, activation="ReLU"):
    layers = [ConvBatchNorm(in_channels, out_channels, activation)]
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class DownBlock(nn.Module):
    """Downsampling by max pooling followed by convolution blocks."""

    def __init__(self, in_channels, out_channels, nb_Conv, activation="ReLU"):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        return self.nConvs(self.maxpool(x))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    """Original channel-wise cross attention in the UCTransNet decoder."""

    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(Flatten(), nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(Flatten(), nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.adaptive_avg_pool2d(x, output_size=1)
        channel_att_x = self.mlp_x(avg_pool_x)

        avg_pool_g = F.adaptive_avg_pool2d(g, output_size=1)
        channel_att_g = self.mlp_g(avg_pool_g)

        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return self.relu(x * scale)


class UpBlockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation="ReLU"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        return self.nConvs(torch.cat([skip_x_att, up], dim=1))


class ChannelAttentionCBAM(nn.Module):
    """CBAM channel attention: AvgPool and MaxPool with a shared MLP."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_descriptor = F.adaptive_avg_pool2d(x, output_size=1)
        max_descriptor = F.adaptive_max_pool2d(x, output_size=1)
        scale = self.sigmoid(
            self.shared_mlp(avg_descriptor) + self.shared_mlp(max_descriptor)
        )
        return x * scale


class SpatialAttentionCBAM(nn.Module):
    """CBAM spatial attention from channel-wise average and maximum maps."""

    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("CBAM spatial kernel size must be 3 or 7")
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True).values
        scale = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Full CBAM: channel attention followed by spatial attention."""

    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttentionCBAM(channels, reduction)
        self.spatial_attention = SpatialAttentionCBAM(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        return self.spatial_attention(x)


class UCTransNet(nn.Module):
    """UCTransNet with full CBAM and without SDI."""

    def __init__(
        self,
        config,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        use_sdi=False,
    ):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Kept only to make the constructor compatible with uctransnet_model.py.
        # This pure UCTransNet+CBAM model never constructs or executes SDI.
        self.use_sdi = False

        base = config.base_channel
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.inc = ConvBatchNorm(n_channels, c1)
        self.down1 = DownBlock(c1, c2, nb_Conv=2)
        self.down2 = DownBlock(c2, c3, nb_Conv=2)
        self.down3 = DownBlock(c3, c4, nb_Conv=2)
        self.down4 = DownBlock(c4, c4, nb_Conv=2)

        self.mtc = ChannelTransformer(
            config,
            vis,
            img_size,
            channel_num=[c1, c2, c3, c4],
            patchSize=config.patch_sizes,
        )

        # Full CBAM is applied independently at every reconstructed skip scale.
        self.cbam1 = CBAM(c1)
        self.cbam2 = CBAM(c2)
        self.cbam3 = CBAM(c3)
        self.cbam4 = CBAM(c4)

        self.up4 = UpBlockAttention(c4 * 2, c3, nb_Conv=2)
        self.up3 = UpBlockAttention(c3 * 2, c2, nb_Conv=2)
        self.up2 = UpBlockAttention(c2 * 2, c1, nb_Conv=2)
        self.up1 = UpBlockAttention(c1 * 2, c1, nb_Conv=2)

        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1, stride=1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4)

        s1 = self.cbam1(x1)
        s2 = self.cbam2(x2)
        s3 = self.cbam3(x3)
        s4 = self.cbam4(x4)

        x = self.up4(x5, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        logits = self.outc(x)
        output = self.last_activation(logits) if self.n_classes == 1 else logits
        return (output, att_weights) if self.vis else output
