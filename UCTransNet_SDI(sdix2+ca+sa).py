# -*- coding: utf-8 -*-
# UCTransNet + SDI (Skip-connection Dynamic Integration)
# - Baseline UCTransNet from official repo (AAAI 2022) with optional SDI fusion on skip features.
# - Add CA/SA gating BEFORE SDI inputs (dynamic weighting), WITHOUT changing SDI itself.

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
    """(conv => BN => ReLU)"""
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = [ConvBatchNorm(in_channels, out_channels, activation)]
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class DownBlock(nn.Module):
    """Downscaling with maxpool then convs"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        return self.nConvs(self.maxpool(x))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    """Channel-wise Cross Attention (CCA)"""
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(Flatten(), nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(Flatten(), nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                  stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)

        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)),
                                  stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)

        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = self.relu(x * scale)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)
        return self.nConvs(x)


class SDI(nn.Module):
    """SDI from UNetV2: multiply multi-scale features after resizing to anchor."""
    def __init__(self, channel: int):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(channel, channel, 3, 1, 1) for _ in range(4)])

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_h, target_w = anchor.shape[-2], anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-2] > target_h or x.shape[-1] > target_w:
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))
            elif x.shape[-2] < target_h or x.shape[-1] < target_w:
                x = F.interpolate(x, size=(target_h, target_w),
                                  mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans


class ChannelAttention(nn.Module):
    """SE-style channel attention: output (B,C,1,1)"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention: output (B,1,H,W)"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)  # (B,2,H,W)
        att = self.conv1(x_out)                       # (B,1,H,W)
        att = self.sigmoid(att)
        return x * att


class UCTransNet(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, img_size=256, vis=False, use_sdi=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_sdi = use_sdi

        base = config.base_channel
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.inc = ConvBatchNorm(n_channels, c1)
        self.down1 = DownBlock(c1, c2, nb_Conv=2)
        self.down2 = DownBlock(c2, c3, nb_Conv=2)
        self.down3 = DownBlock(c3, c4, nb_Conv=2)
        self.down4 = DownBlock(c4, c4, nb_Conv=2)

        self.mtc = ChannelTransformer(
            config, vis, img_size,
            channel_num=[c1, c2, c3, c4],
            patchSize=config.patch_sizes
        )

        # ---- SDI + (CA/SA gating BEFORE SDI)
        if use_sdi:
            # stage 4 (anchor c4)
            self.sdi4 = SDI(c4)
            self.p14 = nn.Conv2d(c1, c4, 1)
            self.p24 = nn.Conv2d(c2, c4, 1)
            self.p34 = nn.Conv2d(c3, c4, 1)
            self.p44 = nn.Identity()
            self.sdi4_ca = ChannelAttention(c4)
            self.sdi4_sa = SpatialAttention()

            # stage 3 (anchor c3)
            self.sdi3 = SDI(c3)
            self.p13 = nn.Conv2d(c1, c3, 1)
            self.p23 = nn.Conv2d(c2, c3, 1)
            self.p33 = nn.Identity()
            self.p43 = nn.Conv2d(c4, c3, 1)
            self.sdi3_ca = ChannelAttention(c3)
            self.sdi3_sa = SpatialAttention()

            # stage 2 (anchor c2)
            self.sdi2 = SDI(c2)
            self.p12 = nn.Conv2d(c1, c2, 1)
            self.p22 = nn.Identity()
            self.p32 = nn.Conv2d(c3, c2, 1)
            self.p42 = nn.Conv2d(c4, c2, 1)
            self.sdi2_ca = ChannelAttention(c2)
            self.sdi2_sa = SpatialAttention()

            # stage 1 (anchor c1)
            self.sdi1 = SDI(c1)
            self.p11 = nn.Identity()
            self.p21 = nn.Conv2d(c2, c1, 1)
            self.p31 = nn.Conv2d(c3, c1, 1)
            self.p41 = nn.Conv2d(c4, c1, 1)
            self.sdi1_ca = ChannelAttention(c1)
            self.sdi1_sa = SpatialAttention()

        self.up4 = UpBlock_attention(c4 * 2, c3, nb_Conv=2)
        self.up3 = UpBlock_attention(c3 * 2, c2, nb_Conv=2)
        self.up2 = UpBlock_attention(c2 * 2, c1, nb_Conv=2)
        self.up1 = UpBlock_attention(c1 * 2, c1, nb_Conv=2)

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

        if self.use_sdi:
            # CA+SA gating BEFORE SDI
            xs4 = [self.sdi4_sa(self.sdi4_ca(self.p14(x1))),
                   self.sdi4_sa(self.sdi4_ca(self.p24(x2))),
                   self.sdi4_sa(self.sdi4_ca(self.p34(x3))),
                   self.sdi4_sa(self.sdi4_ca(self.p44(x4)))]
            s4 = self.sdi4(xs4, x4)
            s4 = self.sdi4(xs4, s4)

            xs3 = [self.sdi3_sa(self.sdi3_ca(self.p13(x1))),
                   self.sdi3_sa(self.sdi3_ca(self.p23(x2))),
                   self.sdi3_sa(self.sdi3_ca(self.p33(x3))),
                   self.sdi3_sa(self.sdi3_ca(self.p43(x4)))]
            s3 = self.sdi3(xs3, x3)
            s3 = self.sdi3(xs3, s3)

            xs2 = [self.sdi2_sa(self.sdi2_ca(self.p12(x1))),
                   self.sdi2_sa(self.sdi2_ca(self.p22(x2))),
                   self.sdi2_sa(self.sdi2_ca(self.p32(x3))),
                   self.sdi2_sa(self.sdi2_ca(self.p42(x4)))]
            s2 = self.sdi2(xs2, x2)
            s2 = self.sdi2(xs2, s2)

            xs1 = [self.sdi1_sa(self.sdi1_ca(self.p11(x1))),
                   self.sdi1_sa(self.sdi1_ca(self.p21(x2))),
                   self.sdi1_sa(self.sdi1_ca(self.p31(x3))),
                   self.sdi1_sa(self.sdi1_ca(self.p41(x4)))]
            s1 = self.sdi1(xs1, x1)
            s1 = self.sdi1(xs1, s1)
        else:
            s4, s3, s2, s1 = x4, x3, x2, x1

        x = self.up4(x5, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        logits = self.last_activation(self.outc(x)) if self.n_classes == 1 else self.outc(x)
        return (logits, att_weights) if self.vis else logits
