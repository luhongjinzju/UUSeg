# losses_recon.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1WithDeepSupervision(nn.Module):
    """
    兼容 2D/3D 的 L1 loss：
    - out 可以是 Tensor 或 List[Tensor] (deep supervision)
    - 自动根据 target 的维度选择 bilinear/trilinear
    """
    def __init__(self, num_outputs: int, weights=None):
        super().__init__()
        self.num_outputs = int(num_outputs)
        if weights is None:
            # 默认：越高分辨率权重越大（也可以均匀）
            # 例如 num_outputs=5 -> [1.0,0.5,0.25,0.125,0.0625]
            self.weights = [1.0 / (2 ** i) for i in range(self.num_outputs)]
        else:
            assert len(weights) == self.num_outputs
            self.weights = list(weights)

    def forward(self, outputs, target):
        """
        outputs:
          - Tensor: (N,C,H,W) or (N,C,D,H,W)
          - list/tuple of Tensor: deep supervision outputs
        target:
          - Tensor: (N,C,H,W) or (N,C,D,H,W)
        """
        if isinstance(outputs, (list, tuple)):
            outs = list(outputs)
        else:
            outs = [outputs]

        # 如果传入的 num_outputs 和实际输出个数不一致，也不直接炸，按实际个数算
        n = min(self.num_outputs, len(outs))
        if n <= 0:
            raise RuntimeError("No outputs for loss computation.")

        # target 维度判断：2D -> 4维 (N,C,H,W)；3D -> 5维 (N,C,D,H,W)
        if target.ndim == 4:
            mode = "bilinear"
            align = False
            get_size = lambda o: o.shape[-2:]  # (H,W)
        elif target.ndim == 5:
            mode = "trilinear"
            align = False
            get_size = lambda o: o.shape[-3:]  # (D,H,W)
        else:
            raise ValueError(f"Unsupported target ndim={target.ndim}, expected 4 or 5.")

        loss = 0.0
        weight_sum = 0.0

        for i in range(n):
            out = outs[i]
            # 将 target resize 到 out 的空间分辨率（维度一致）
            tgt = F.interpolate(target, size=get_size(out), mode=mode, align_corners=align)
            w = float(self.weights[i]) if i < len(self.weights) else 1.0
            loss = loss + w * F.l1_loss(out, tgt)
            weight_sum += w

        # 归一化一下，避免 weights 改变 loss 尺度太大
        loss = loss / max(weight_sum, 1e-8)
        return loss
