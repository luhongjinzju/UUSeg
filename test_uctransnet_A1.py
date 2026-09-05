import os
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff

from uctransnet_model import build_uctransnet

# =========================
# 测试集路径（保持你的数据集不变）
# =========================
test_img_dir  = r"D:\项目unet\寒假\my_code\val_dataset\reconstruction_val"
test_mask_dir = r"D:\项目unet\寒假\my_code\val_dataset\gold_mask_val"

# =========================
# 输出路径：全部切换到 uctransnet_A
# =========================
proj_root = r"D:\项目unet\寒假\my_code\uctransnet_A"
save_pred_dir = os.path.join(proj_root, "pred_best")
os.makedirs(save_pred_dir, exist_ok=True)


# 选择要测试的 ckpt（baseline or SDI）
USE_SDI = True
ckpt_name = "UCTNet_SDI_best.pth" if USE_SDI else "UCTNet_best.pth"
ckpt_path = os.path.join(proj_root, "results", ckpt_name)

IN_NORM_DEN = 4095.0
MASK_DEN    = 255.0
IMG_SIZE    = 256

# 统一阈值：IoU/Dice/PA/ASD 全部使用这个阈值
THR = 0.65

# 每隔多少张保存一次可视化（如果你想全部保存就设为1）
SAVE_EVERY = 1


def list_common_files(img_dir, mask_dir):
    img_set = set([os.path.splitext(f)[0] for f in os.listdir(img_dir)
                   if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))])
    msk_set = set([os.path.splitext(f)[0] for f in os.listdir(mask_dir)
                   if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))])
    return sorted(list(img_set & msk_set))


def read_any(path, size=256, is_mask=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = tiff.imread(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    img = cv2.resize(img, (size, size), interpolation=interp)
    return img


class SegPairDataset(Dataset):
    def __init__(self, img_dir, mask_dir, names, size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.names = names
        self.size = size

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
            raise FileNotFoundError(base)

        img = read_any(img_path, self.size, is_mask=False).astype(np.float32)
        mask = read_any(mask_path, self.size, is_mask=True).astype(np.float32)

        # 归一化
        img_norm = np.clip(img / IN_NORM_DEN, 0, 1)

        # GT mask 归一化并二值化
        mask = np.clip(mask / MASK_DEN, 0, 1)
        mask = (mask > 0.5).astype(np.float32)

        img_t = torch.from_numpy(img_norm).unsqueeze(0)    # (1,H,W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)       # (1,H,W)

        # 额外返回原始 img（用于可视化），避免反归一化误差
        return img_t, mask_t, base, img.astype(np.float32)


def metrics_iou_dice_pa(gt, pr, thr=0.5):
    gt = (gt >= thr)
    pr = (pr >= thr)

    TP = np.sum(gt & pr)
    TN = np.sum(~gt & ~pr)
    FP = np.sum(~gt & pr)
    FN = np.sum(gt & ~pr)

    pa = (TP + TN) / (TP + TN + FP + FN + 1e-12)
    iou = TP / (TP + FP + FN + 1e-12)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-12)
    return float(pa), float(iou), float(dice)


def _boundary(mask_bin):
    mask_u8 = (mask_bin.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    er = cv2.erode(mask_u8, kernel, iterations=1)
    bd = (mask_u8 > 0) & (er == 0)
    return bd.astype(np.uint8)


def asd_surface_distance(gt_bin, pr_bin):
    gt_bd = _boundary(gt_bin)
    pr_bd = _boundary(pr_bin)

    gt_n = int(gt_bd.sum())
    pr_n = int(pr_bd.sum())
    if gt_n == 0 and pr_n == 0:
        return 0.0
    if gt_n == 0 or pr_n == 0:
        return float("nan")

    inv_gt = (1 - gt_bd).astype(np.uint8)
    inv_pr = (1 - pr_bd).astype(np.uint8)

    dt_to_gt = cv2.distanceTransform(inv_gt, distanceType=cv2.DIST_L2, maskSize=3)
    dt_to_pr = cv2.distanceTransform(inv_pr, distanceType=cv2.DIST_L2, maskSize=3)

    d_pr_to_gt = dt_to_gt[pr_bd.astype(bool)].mean()
    d_gt_to_pr = dt_to_pr[gt_bd.astype(bool)].mean()

    return float((d_pr_to_gt + d_gt_to_pr) / 2.0)


def save_visuals(base, img_raw, gt_bin, prob, pred_bin, out_dir):
    """
    img_raw: float32 (H,W) 原始强度（未归一化）
    gt_bin, pred_bin: bool or {0,1} (H,W)
    prob: float32 (H,W) 0~1
    """
    os.makedirs(out_dir, exist_ok=True)

    # # ---- 1) 保存概率图（16-bit tif）
    # prob16 = np.clip(prob * 65535.0, 0, 65535).astype(np.uint16)
    # tiff.imwrite(os.path.join(out_dir, f"{base}_pred_prob.tif"), prob16)

    # ---- 2) 保存二值 mask（0/255 tif）
    pred_u8 = (pred_bin.astype(np.uint8) * 255)
    tiff.imwrite(os.path.join(out_dir, f"{base}.tif"), pred_u8)

    # # ---- 3) overlay：在输入上画预测轮廓（红）和GT轮廓（绿）
    # img_norm8 = img_raw.copy()
    # # 拉伸到 0~255 便于显示（防止过暗）
    # vmin, vmax = np.percentile(img_norm8, 1), np.percentile(img_norm8, 99)
    # img_norm8 = np.clip((img_norm8 - vmin) / (vmax - vmin + 1e-12), 0, 1)
    # img_u8 = (img_norm8 * 255).astype(np.uint8)
    # overlay = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    #
    # pred_cnts, _ = cv2.findContours(pred_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # gt_u8 = (gt_bin.astype(np.uint8) * 255)
    # gt_cnts, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # cv2.drawContours(overlay, pred_cnts, -1, (0, 0, 255), 1)  # 红：Pred
    # cv2.drawContours(overlay, gt_cnts,   -1, (0, 255, 0), 1)  # 绿：GT
    # cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), overlay)
    #
    # # ---- 4) 三联图：Input / GT / Pred
    # gt_vis = gt_u8
    # pr_vis = pred_u8
    # comp = np.concatenate([img_u8, gt_vis, pr_vis], axis=1)
    # cv2.imwrite(os.path.join(out_dir, f"{base}_compare.png"), comp)


def _model_forward(model, img_t):
    """兼容 model 返回 (pred, att) 的情况"""
    pred = model(img_t)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    return pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("USE_SDI =", USE_SDI)
    print("CKPT:", ckpt_path)
    print("THR =", THR)

    names = list_common_files(test_img_dir, test_mask_dir)
    print("Found pairs:", len(names))
    if len(names) == 0:
        raise RuntimeError("测试集为空或无法配对：检查 test_img_dir/test_mask_dir 文件名是否一致")

    ds = SegPairDataset(test_img_dir, test_mask_dir, names, size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_uctransnet(use_sdi=USE_SDI).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    out_csv = os.path.join(save_pred_dir, "metrics.csv")
    rows = []

    with torch.no_grad():
        for idx, (img_t, mask_t, base, img_raw) in enumerate(loader, start=1):
            img_t = img_t.to(device).float()
            mask_t = mask_t.to(device).float()

            pred = _model_forward(model, img_t)  # (1,1,H,W) after sigmoid in your model
            pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float32)
            gt_np = mask_t.squeeze().detach().cpu().numpy().astype(np.float32)

            pa, iou, dice = metrics_iou_dice_pa(gt_np, pred_np, thr=THR)

            # ASD 使用同一个阈值（修复你原来的 0.5）
            gt_bin = (gt_np >= THR)
            pr_bin = (pred_np >= THR)
            asd = asd_surface_distance(gt_bin, pr_bin)

            rows.append([base[0] if isinstance(base, (list, tuple)) else base, pa, iou, dice, asd])

            # 保存可视化
            if SAVE_EVERY > 0 and ((idx - 1) % SAVE_EVERY == 0):
                base_name = base[0] if isinstance(base, (list, tuple)) else base
                img_raw_np = img_raw.squeeze().numpy()
                save_visuals(
                    base=base_name,
                    img_raw=img_raw_np,
                    gt_bin=gt_bin,
                    prob=pred_np,
                    pred_bin=pr_bin,
                    out_dir=save_pred_dir
                )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "PA", "IoU", "Dice", "ASD"])
        w.writerows(rows)

    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=np.float64)
    mean_pa = np.nanmean(arr[:, 0])
    mean_iou = np.nanmean(arr[:, 1])
    mean_dice = np.nanmean(arr[:, 2])
    mean_asd = np.nanmean(arr[:, 3])

    print("Mean PA/IoU/Dice/ASD:", mean_pa, mean_iou, mean_dice, mean_asd)
    print("Saved metrics:", out_csv)
    print("Saved images to:", save_pred_dir)


if __name__ == "__main__":
    main()
