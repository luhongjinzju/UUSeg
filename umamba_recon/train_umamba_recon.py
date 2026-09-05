import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_recon import PairedTifDataset
from model_umamba_recon import build_umamba_recon, is_3d_net
from losses_recon import L1WithDeepSupervision
from utils_recon import seed_everything, save_checkpoint


# =======================
# 路径
# =======================

# 训练集根目录：你已经确认存在 input/ fluorescence/
TRAIN_ROOT = Path("/home/crx/home/crx/pycharm_projects/my_code/temp_dataset")
TRAIN_INPUT_SUBDIR = "input"
TRAIN_TARGET_SUBDIR = "fluorescence"

# 验证集根目录：你之前一直用 test_dataset/input_test + fluorescence_test
VAL_ROOT = Path("/home/crx/home/crx/pycharm_projects/my_code/test_dataset")
VAL_INPUT_SUBDIR = "input_test"
VAL_TARGET_SUBDIR = "fluorescence_test"

# 输出根目录（写死），并按 net 自动分子目录
BASE_OUT_DIR = Path("/home/crx/home/crx/pycharm_projects/my_code/umamba_recon/experiments")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=str, default="Enc2d",
                   choices=["Enc2d", "Bot2d", "Enc3d", "Bot3d"],
                   help="选择 U-Mamba 网络结构")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--in_max", type=float, default=4095.0)
    p.add_argument("--out_max", type=float, default=4095.0)

    p.add_argument("--no_ds", action="store_true", help="关闭 deep supervision")

    # 每 N epoch 额外保存一次 epoch_*.pth
    p.add_argument("--save_every", type=int, default=10)

    # 2D->3D 堆叠规则（仅 3D 网络会用到）
    p.add_argument("--stack_depth", type=int, default=5)
    p.add_argument("--stack_stride", type=int, default=1)

    return p.parse_args()


def _assert_dir(p: Path, name: str):
    if not p.exists():
        raise FileNotFoundError(f"[PathError] {name} not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"[PathError] {name} is not a directory: {p}")


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial_dim = 3 if is_3d_net(args.net) else 2

    # ✅ 按 net 分子目录，避免四网络互相覆盖
    out_dir = BASE_OUT_DIR / args.net
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    # ✅ 拼训练/验证真实目录
    train_input_dir = TRAIN_ROOT / TRAIN_INPUT_SUBDIR
    train_target_dir = TRAIN_ROOT / TRAIN_TARGET_SUBDIR

    val_input_dir = VAL_ROOT / VAL_INPUT_SUBDIR
    val_target_dir = VAL_ROOT / VAL_TARGET_SUBDIR

    # ✅ 防呆检查：一开始就报清楚路径问题，不会跑半天才报错
    _assert_dir(TRAIN_ROOT, "TRAIN_ROOT")
    _assert_dir(VAL_ROOT, "VAL_ROOT")
    _assert_dir(train_input_dir, "train_input_dir")
    _assert_dir(train_target_dir, "train_target_dir")
    _assert_dir(val_input_dir, "val_input_dir")
    _assert_dir(val_target_dir, "val_target_dir")

    print("========== Dataset Paths ==========")
    print("train_input :", train_input_dir)
    print("train_target:", train_target_dir)
    print("val_input   :", val_input_dir)
    print("val_target  :", val_target_dir)
    print("spatial_dim :", spatial_dim, "| net:", args.net)
    print("===================================")

    # ✅ 数据集：2D 或 2D->3D 堆叠（3D网络才生效）
    train_ds = PairedTifDataset(
        str(train_input_dir), str(train_target_dir),
        in_max=args.in_max, out_max=args.out_max,
        augment=True,
        spatial_dim=spatial_dim,
        stack_depth=args.stack_depth,
        stack_stride=args.stack_stride,
    )
    val_ds = PairedTifDataset(
        str(val_input_dir), str(val_target_dir),
        in_max=args.in_max, out_max=args.out_max,
        augment=False,
        spatial_dim=spatial_dim,
        stack_depth=args.stack_depth,
        stack_stride=args.stack_stride,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    input_size_3d = (args.stack_depth, 256, 256) if is_3d_net(args.net) else None

    model = build_umamba_recon(
        args.net,
        in_channels=1,
        out_channels=1,
        deep_supervision=(not args.no_ds),
        input_size_2d=(256, 256),
        input_size_3d=input_size_3d,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # deep supervision 打开时，UMamba 通常会输出多尺度列表；关闭时输出单个 tensor
    num_outputs = 3 if (not args.no_ds) else 1
    criterion = L1WithDeepSupervision(num_outputs=num_outputs)

    best = 1e9

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"[{args.net}] Epoch {epoch+1}/{args.epochs}", ncols=110)

        train_losses = []
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        train_l1 = float(sum(train_losses) / max(1, len(train_losses)))
        val_l1 = validate(model, val_loader, criterion, device)

        # ✅ last：每 epoch 覆盖
        save_checkpoint(str(out_dir / "last.pth"), model, optimizer, epoch, best)

        # ✅ epoch_k：每 save_every 保存一次（不会覆盖，因为名字不同）
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(str(out_dir / f"epoch_{epoch+1}.pth"), model, optimizer, epoch, best)

        # ✅ best：val 最优覆盖
        if val_l1 < best:
            best = val_l1
            save_checkpoint(str(out_dir / "best.pth"), model, optimizer, epoch, best)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"epoch={epoch+1}, train_l1={train_l1:.6f}, val_l1={val_l1:.6f}, best={best:.6f}\n")

        print(f"[{args.net}] epoch={epoch+1} train_l1={train_l1:.6f} val_l1={val_l1:.6f} best={best:.6f}")

    print(f"✅ Done. Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
