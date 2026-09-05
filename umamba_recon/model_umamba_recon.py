# model_umamba_recon.py
from __future__ import annotations

from typing import Tuple, Optional
import inspect
import torch
from torch import nn


def is_3d_net(net: str) -> bool:
    return net.lower().endswith("3d")


def forward_get_highres(out):
    """兼容 deep supervision：取最高分辨率输出。"""
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def _construct_filter_kwargs(cls, cfg: dict):
    """
    关键：UMambaEnc 与 UMambaBot 的 __init__ 参数不一致。
    对 Bot 系列，我们根据签名过滤 cfg，只传它支持的参数，避免 unexpected keyword。
    """
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    filtered = {k: v for k, v in cfg.items() if k in allowed}
    return cls(**filtered)


def _default_unet_config_2d(
    input_size: Tuple[int, int] = (256, 256),
    input_channels: int = 1,
    num_classes: int = 1,
    deep_supervision: bool = True,
):
    n_stages = 5
    features_per_stage = [32, 64, 128, 256, 320]
    conv_op = nn.Conv2d
    kernel_sizes = [(3, 3)] * n_stages
    strides = [(1, 1), (2, 2), (2, 2), (2, 2), (2, 2)]
    n_conv_per_stage = [2] * n_stages
    n_conv_per_stage_decoder = [2] * (n_stages - 1)

    return dict(
        input_size=input_size,
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,

        # ✅ 必须是 dict，不能 None
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},

        deep_supervision=deep_supervision,
    )


def _default_unet_config_3d(
    input_size: Tuple[int, int, int],
    input_channels: int = 1,
    num_classes: int = 1,
    deep_supervision: bool = True,
):
    n_stages = 5
    features_per_stage = [32, 64, 128, 256, 320]
    conv_op = nn.Conv3d
    kernel_sizes = [(3, 3, 3)] * n_stages

    # 只在 H/W 下采样，不在 D 下采样（D=stack_depth）
    strides = [(1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
    n_conv_per_stage = [2] * n_stages
    n_conv_per_stage_decoder = [2] * (n_stages - 1)

    return dict(
        input_size=input_size,
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,

        # ✅ 必须是 dict，不能 None
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},

        deep_supervision=deep_supervision,
    )


def build_umamba_recon(
    net: str,
    in_channels: int = 1,
    out_channels: int = 1,
    deep_supervision: bool = True,
    input_size_2d: Tuple[int, int] = (256, 256),
    input_size_3d: Optional[Tuple[int, int, int]] = None,
) -> nn.Module:
    net = net.strip()

    if net == "Enc2d":
        from nnunetv2.nets.UMambaEnc_2d import UMambaEnc
        cfg = _default_unet_config_2d(
            input_size=input_size_2d,
            input_channels=in_channels,
            num_classes=out_channels,
            deep_supervision=deep_supervision,
        )
        return UMambaEnc(**cfg)

    if net == "Bot2d":
        from nnunetv2.nets.UMambaBot_2d import UMambaBot
        cfg = _default_unet_config_2d(
            input_size=input_size_2d,
            input_channels=in_channels,
            num_classes=out_channels,
            deep_supervision=deep_supervision,
        )
        # ✅ 关键：Bot2d 不一定支持 input_size 等参数 -> 过滤
        return _construct_filter_kwargs(UMambaBot, cfg)

    if net == "Enc3d":
        if input_size_3d is None:
            raise ValueError("Enc3d requires input_size_3d=(D,H,W), D must equal stack_depth.")
        from nnunetv2.nets.UMambaEnc_3d import UMambaEnc
        cfg = _default_unet_config_3d(
            input_size=input_size_3d,
            input_channels=in_channels,
            num_classes=out_channels,
            deep_supervision=deep_supervision,
        )
        return UMambaEnc(**cfg)

    if net == "Bot3d":
        if input_size_3d is None:
            raise ValueError("Bot3d requires input_size_3d=(D,H,W), D must equal stack_depth.")
        from nnunetv2.nets.UMambaBot_3d import UMambaBot
        cfg = _default_unet_config_3d(
            input_size=input_size_3d,
            input_channels=in_channels,
            num_classes=out_channels,
            deep_supervision=deep_supervision,
        )
        # ✅ 关键：Bot3d 也做过滤
        return _construct_filter_kwargs(UMambaBot, cfg)

    raise ValueError(f"Unknown net={net}. choices: Enc2d Bot2d Enc3d Bot3d")
