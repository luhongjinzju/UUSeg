import os, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tifffile

from uctransnet_model import build_uctransnet
from utils import WeightedDiceBCE

# =========================
# 数据路径
# =========================
train_images_path = r"D:/项目unet/my_code/temp_dataset/reconstruction"
train_masks_path  = r"D:/项目unet/my_code/temp_dataset/mask"

val_img_dir  = r"D:/项目unet/my_code/test_dataset/reconstruction_test"
val_mask_dir = r"D:/项目unet/my_code/test_dataset/gold_mask_test"

# =========================
# 输出路径：全部切换到 uctransnet_A
# =========================
proj_root = r"D:/项目unet/my_code/uctransnet_A"
save_dir = os.path.join(proj_root, "results")
os.makedirs(save_dir, exist_ok=True)

# =========================
# 超参
# =========================
img_size = 256
batch_size = 4
epochs = 500
lr = 1e-4
num_workers = 4
seed = 666

# 归一化
IN_NORM_DEN = 4095.0
MASK_DEN    = 255.0

# 是否启用 SDI（默认 False = 严格 baseline；True = baseline + SDI）
USE_SDI = True


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def list_common_files(img_dir, mask_dir):
    img_set = set([os.path.splitext(f)[0] for f in os.listdir(img_dir)
                   if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))])
    msk_set = set([os.path.splitext(f)[0] for f in os.listdir(mask_dir)
                   if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))])
    return sorted(list(img_set & msk_set))


def read_any(path, size=256, is_mask=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = tifffile.imread(path)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    img = cv2.resize(img, (size, size), interpolation=interp)
    return img


def random_rot_flip(img, mask):
    k = np.random.randint(0, 4)
    img = np.rot90(img, k).copy()
    mask = np.rot90(mask, k).copy()
    if np.random.rand() > 0.5:
        img = np.flip(img, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()
    if np.random.rand() > 0.5:
        img = np.flip(img, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    return img, mask


class SegPairDataset(Dataset):
    def __init__(self, img_dir, mask_dir, names, size=256, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.names = names
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        base = self.names[idx]

        def find_file(d, base):
            for ext in [".tif", ".tiff", ".png", ".jpg"]:
                p = os.path.join(d, base + ext)
                if os.path.exists(p):
                    return p
            for f in os.listdir(d):
                if os.path.splitext(f)[0] == base:
                    return os.path.join(d, f)
            return None

        img_path = find_file(self.img_dir, base)
        mask_path = find_file(self.mask_dir, base)
        if img_path is None or mask_path is None:
            raise FileNotFoundError(f"Missing pair for {base}")

        img = read_any(img_path, self.size, is_mask=False).astype(np.float32)
        mask = read_any(mask_path, self.size, is_mask=True).astype(np.float32)

        if self.augment:
            img, mask = random_rot_flip(img, mask)

        img = np.clip(img / IN_NORM_DEN, 0, 1)
        mask = np.clip(mask / MASK_DEN, 0, 1)
        mask = (mask > 0.5).astype(np.float32)

        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return {"image": img_t, "label": mask_t}, base


def run_one_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n = 0

    for batch, _ in loader:
        images = batch["image"].to(device, non_blocking=True).float()
        labels = batch["label"].to(device, non_blocking=True).float()

        if is_train:
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = model(images)
                loss = criterion(preds, labels)

        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(1, n)


def main():
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("USE_SDI =", USE_SDI)

    train_names = list_common_files(train_images_path, train_masks_path)
    val_names   = list_common_files(val_img_dir, val_mask_dir)

    if len(train_names) == 0:
        raise RuntimeError("训练集为空：请检查 train_images_path/train_masks_path")
    if len(val_names) == 0:
        raise RuntimeError("验证集为空：请检查 val_img_dir/val_mask_dir")

    print(f"Train: {len(train_names)} | Val: {len(val_names)}")

    train_ds = SegPairDataset(train_images_path, train_masks_path, train_names, img_size, augment=True)
    val_ds   = SegPairDataset(val_img_dir, val_mask_dir, val_names, img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False)

    model = build_uctransnet(use_sdi=USE_SDI).to(device)
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_epoch = -1
    history = []

    log_csv = os.path.join(save_dir, "loss_history.csv")
    best_info = os.path.join(save_dir, "best_info.txt")

    ckpt_name = "UCTNet_SDI_best.pth" if USE_SDI else "UCTNet_best.pth"
    ckpt_path = os.path.join(save_dir, ckpt_name)

    for ep in range(1, epochs + 1):
        tr_loss = run_one_epoch(model, train_loader, device, criterion, optimizer=optimizer)
        va_loss = run_one_epoch(model, val_loader, device, criterion, optimizer=None)

        print(f"[Epoch {ep:03d}/{epochs}] train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")

        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        pd.DataFrame(history).to_csv(log_csv, index=False)

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = ep
            torch.save(model.state_dict(), ckpt_path)
            with open(best_info, "w", encoding="utf-8") as f:
                f.write(f"best_epoch={best_epoch}\n")
                f.write(f"best_val_loss={best_val:.10f}\n")
                f.write(f"use_sdi={USE_SDI}\n")
            print(f"  ==> saved best: {os.path.basename(ckpt_path)} (epoch={best_epoch}, val_loss={best_val:.6f})")

    print("Done.")
    print("Best epoch:", best_epoch, "best val_loss:", best_val)
    print("Log saved:", log_csv)
    print("Best info:", best_info)
    print("Best ckpt:", ckpt_path)


if __name__ == "__main__":
    main()
