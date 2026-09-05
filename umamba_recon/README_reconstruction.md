# U-Mamba (官方网络) —— 2D 明场→荧光 重建（最小改动版）

> 说明：U‑Mamba 官方仓库基于 nnU-Net v2，默认任务是**分割**（分类/离散标签），而“明场→荧光重建”是**回归**（连续像素值）。
> 
> 为了“尽量不改官方结构”，本方案**直接复用官方的 UMambaEnc_2d 网络实现**，但用一个独立的、简洁的 PyTorch 训练/测试脚本来做回归训练。

## 1. 依赖

- Python 3.10
- PyTorch 2.x
- mamba-ssm / causal-conv1d
- tifffile

```bash
pip install torch torchvision
pip install causal-conv1d>=1.2.0 mamba-ssm --no-cache-dir
pip install tifffile numpy tqdm
```

> 你需要先把官方 U-Mamba 仓库安装为可 import（见下方第 2 节）。

## 2. 把官方 U-Mamba 安装为包

在官方仓库根目录（包含 umamba/ 的目录）执行：

```bash
cd U-Mamba-main/umamba
pip install -e .
```

## 3. 数据组织（最简单的配对方式）

假设你有两份 256×256 的 tif：

```
DATA_ROOT/
  input/            # 明场
    0001.tif
    0002.tif
    ...
  target/           # 荧光
    0001.tif
    0002.tif
    ...
```

**要求：文件名一一对应**。

## 4. 训练

```bash
python train_umamba_recon.py \
  --data_root D:/path/to/DATA_ROOT \
  --out_dir   D:/path/to/umamba_recon_runs \
  --epochs 200 \
  --batch_size 8 \
  --lr 1e-4 \
  --in_max 4095 \
  --out_max 4095
```

## 5. 测试/推理

```bash
python test_umamba_recon.py \
  --ckpt D:/path/to/umamba_recon_runs/best.pth \
  --input_dir D:/path/to/DATA_ROOT/input \
  --out_dir   D:/path/to/preds \
  --in_max 4095 \
  --out_max 4095
```

输出默认保存为 `.tif`（uint16），数值范围按 `out_max` 反归一化。
