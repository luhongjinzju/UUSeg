# datasets_recon.py
from __future__ import annotations
import os
from typing import List, Tuple, Optional

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset


def _list_tif_names(folder: str) -> List[str]:
    exts = {".tif", ".tiff"}
    names = [f for f in os.listdir(folder) if os.path.splitext(f.lower())[1] in exts]
    names.sort(key=lambda x: (len(os.path.splitext(x)[0]), os.path.splitext(x)[0], x))
    return names


def _read_tif(path: str) -> np.ndarray:
    arr = tiff.imread(path)
    return arr
def list_tifs_sorted(folder: str) -> List[str]:
    """给测试脚本用：返回排序后的 tif 文件名列表（不含路径）"""
    return _list_tif_names(folder)


def read_tif_2d(path: str) -> np.ndarray:
    """给测试脚本用：读 2D tif，返回 (H,W)。如果读到 3D/多通道就报错，避免 silent bug。"""
    arr = _read_tif(path)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D tif (H,W), got shape={arr.shape} from {path}")
    return arr


class PairedTifDataset(Dataset):
    """
    2D:
      x: (1,H,W), y:(1,H,W)
    2D->3D 堆叠 (volume-to-volume):
      x: (1,D,H,W), y:(1,D,H,W)
    """
    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        in_max: float = 4095.0,
        out_max: float = 4095.0,
        augment: bool = True,
        spatial_dim: int = 2,
        stack_depth: int = 5,
        stack_stride: int = 1,
    ):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.in_max = float(in_max)
        self.out_max = float(out_max)
        self.augment = augment
        self.spatial_dim = int(spatial_dim)
        self.stack_depth = int(stack_depth)
        self.stack_stride = int(stack_stride)

        if self.spatial_dim not in (2, 3):
            raise ValueError("spatial_dim must be 2 or 3")

        # 取交集，保证同名配对
        in_names = _list_tif_names(input_dir)
        tg_names = set(_list_tif_names(target_dir))
        in_names = [n for n in in_names if n in tg_names]
        if len(in_names) == 0:
            raise RuntimeError(f"No paired tif found. input={input_dir}, target={target_dir}")

        self.in_names = in_names

        # 生成索引：2D -> 每张一条；3D -> 滑窗堆叠
        if self.spatial_dim == 2:
            self.indices = list(range(len(self.in_names)))
        else:
            D = self.stack_depth
            if D <= 1:
                raise ValueError("For spatial_dim=3, stack_depth must be >=2")
            stride = max(1, self.stack_stride)
            # 滑窗：0..N-D
            self.indices = list(range(0, len(self.in_names) - D + 1, stride))
            if len(self.indices) == 0:
                raise RuntimeError(f"Not enough slices for stacking: N={len(self.in_names)}, D={D}")

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _rand_flip_rot(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # x,y: (H,W) or (D,H,W)
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=-1)
            y = np.flip(y, axis=-1)
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=-2)
            y = np.flip(y, axis=-2)
        k = np.random.randint(0, 4)
        if k > 0:
            x = np.rot90(x, k, axes=(-2, -1))
            y = np.rot90(y, k, axes=(-2, -1))
        return x.copy(), y.copy()

    def __getitem__(self, idx: int):
        base_i = self.indices[idx]

        if self.spatial_dim == 2:
            name = self.in_names[base_i]
            ip = os.path.join(self.input_dir, name)
            tp = os.path.join(self.target_dir, name)
            x = _read_tif(ip).astype(np.float32)   # (H,W)
            y = _read_tif(tp).astype(np.float32)   # (H,W)

            x = np.clip(x / self.in_max, 0.0, 1.0)
            y = np.clip(y / self.out_max, 0.0, 1.0)

            if self.augment:
                x, y = self._rand_flip_rot(x, y)

            # (1,H,W)
            x = torch.from_numpy(x[None])
            y = torch.from_numpy(y[None])
            return x, y

        # spatial_dim == 3: 2D->3D 堆叠
        D = self.stack_depth
        names = self.in_names[base_i: base_i + D]

        xs = []
        ys = []
        for n in names:
            ip = os.path.join(self.input_dir, n)
            tp = os.path.join(self.target_dir, n)
            xs.append(_read_tif(ip).astype(np.float32))  # (H,W)
            ys.append(_read_tif(tp).astype(np.float32))  # (H,W)

        x = np.stack(xs, axis=0)  # (D,H,W)
        y = np.stack(ys, axis=0)  # (D,H,W)

        x = np.clip(x / self.in_max, 0.0, 1.0)
        y = np.clip(y / self.out_max, 0.0, 1.0)

        if self.augment:
            x, y = self._rand_flip_rot(x, y)

        # (1,D,H,W)
        x = torch.from_numpy(x[None])
        y = torch.from_numpy(y[None])
        return x, y
