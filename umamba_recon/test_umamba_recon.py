# test_umamba_recon.py
import argparse
from pathlib import Path
import re

import numpy as np
import torch
import tifffile as tiff
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from model_umamba_recon import build_umamba_recon, is_3d_net, forward_get_highres
from utils_recon import seed_everything


# ================== 路径写死（与你当前工程一致） ==================
TEST_ROOT = Path("/home/crx/home/crx/pycharm_projects/寒假/my_code/test_dataset")
INPUT_DIR = TEST_ROOT / "input_test"
GT_DIR = TEST_ROOT / "fluorescence_test"

EXP_ROOT = Path("/home/crx/home/crx/pycharm_projects/寒假/my_code/umamba_recon/experiments")

IN_MAX = 4095.0
OUT_MAX = 4095.0
INPUT_SIZE_2D = (256, 256)
# ==================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--net", required=True, choices=["Enc2d", "Bot2d", "Enc3d", "Bot3d"])
    p.add_argument("--stack_depth", type=int, default=5, help="3D堆叠深度D（2D时忽略）")
    p.add_argument("--stack_stride", type=int, default=1, help="3D滑窗步长（2D时忽略）")
    p.add_argument("--seed", type=int, default=42)

    # 遍历所有 ckpt 时可筛选
    p.add_argument(
        "--ckpt_regex",
        type=str,
        default="",
        help="只评估匹配该正则的ckpt名，例如 'epoch_.*|best' 或 'best|last|epoch_100'",
    )

    # 保存预测
    p.add_argument("--save_preds", action="store_true", help="保存每个ckpt的预测结果")
    p.add_argument("--save_uint16", action="store_true", help="保存uint16 tif（默认float32）")

    return p.parse_args()


def list_checkpoints(net_dir: Path):
    ckpts = list(net_dir.glob("epoch_*.pth"))
    if (net_dir / "best.pth").exists():
        ckpts.append(net_dir / "best.pth")
    if (net_dir / "last.pth").exists():
        ckpts.append(net_dir / "last.pth")
    return sorted(ckpts, key=lambda p: p.name)


def list_tifs(folder: Path):
    exts = {".tif", ".tiff"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda x: x.name)
    return files


def _read_2d_tif_float32(path: Path) -> np.ndarray:
    arr = tiff.imread(str(path))
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D tif, got shape={arr.shape} from {path}")
    return arr.astype(np.float32)


def make_stack_by_files(files, start_idx: int, depth: int) -> np.ndarray:
    """files: List[Path]，堆叠 depth 张 2D tif -> (D,H,W)"""
    stack = [_read_2d_tif_float32(files[start_idx + k]) for k in range(depth)]
    return np.stack(stack, axis=0).astype(np.float32)


def metrics_2d(pred2d: np.ndarray, gt2d: np.ndarray):
    """pred2d, gt2d: (H,W) in intensity scale (0..OUT_MAX)"""
    mae = float(np.mean(np.abs(pred2d - gt2d)))
    rmse = float(np.sqrt(np.mean((pred2d - gt2d) ** 2)))
    nrmse = float(rmse / (gt2d.max() - gt2d.min() + 1e-8))
    psnr = float(20 * np.log10(OUT_MAX) - 10 * np.log10(np.mean((pred2d - gt2d) ** 2) + 1e-12))
    ssim_v = float(ssim(pred2d, gt2d, data_range=OUT_MAX))
    return psnr, ssim_v, mae, nrmse


def metrics_3d(pred: np.ndarray, gt: np.ndarray):
    """
    pred, gt: (D,H,W) in intensity scale (0..OUT_MAX)
    SSIM：按 slice 逐张算，再取均值（skimage 的 ssim 对 2D 最稳）
    其它指标：按 volume 直接算
    """
    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    nrmse = float(rmse / (gt.max() - gt.min() + 1e-8))
    psnr = float(20 * np.log10(OUT_MAX) - 10 * np.log10(np.mean((pred - gt) ** 2) + 1e-12))

    ssims = []
    d = pred.shape[0]
    for i in range(d):
        ssims.append(float(ssim(pred[i], gt[i], data_range=OUT_MAX)))
    ssim_v = float(np.mean(ssims)) if len(ssims) else 0.0

    return psnr, ssim_v, mae, nrmse


def _save_pred_tif(path: Path, arr: np.ndarray, save_uint16: bool):
    if save_uint16:
        out = np.clip(np.round(arr), 0, 65535).astype(np.uint16)
    else:
        out = arr.astype(np.float32)
    tiff.imwrite(str(path), out)


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {INPUT_DIR}")
    if not GT_DIR.exists():
        raise FileNotFoundError(f"GT dir not found: {GT_DIR}")

    net_dir = EXP_ROOT / args.net
    if not net_dir.exists():
        raise FileNotFoundError(f"Net dir not found: {net_dir}")

    ckpts = list_checkpoints(net_dir)
    if args.ckpt_regex:
        pat = re.compile(args.ckpt_regex)
        ckpts = [c for c in ckpts if pat.search(c.name)]
    if len(ckpts) == 0:
        raise RuntimeError(f"No checkpoints found to evaluate in {net_dir}")

    files_in = list_tifs(INPUT_DIR)
    if len(files_in) == 0:
        raise RuntimeError(f"No tif files found in {INPUT_DIR}")

    # 为了确保 GT 对齐，按“同名文件”构造 gt 路径列表
    # 3D 堆叠时也用同样的排序与窗口起点，保证输入/GT一致
    files_gt = [GT_DIR / f.name for f in files_in]
    missing_gt = [p.name for p in files_gt if not p.exists()]
    if missing_gt:
        raise RuntimeError(f"Missing {len(missing_gt)} GT files in {GT_DIR}, e.g. {missing_gt[:10]}")

    spatial_dim = 3 if is_3d_net(args.net) else 2
    input_size_3d = (args.stack_depth, 256, 256) if spatial_dim == 3 else None

    # preds 根目录（按 net）
    preds_root = net_dir / "preds_ckpt"
    if args.save_preds:
        preds_root.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"Evaluating {args.net} | {len(ckpts)} checkpoints")
    print(f"save_preds={args.save_preds} save_uint16={args.save_uint16}")
    if spatial_dim == 3:
        print(f"stack_depth={args.stack_depth} stack_stride={args.stack_stride}")

    for ckpt in ckpts:
        print(f"\n>>> {ckpt.name}")

        # 建模
        model = build_umamba_recon(
            args.net,
            in_channels=1,
            out_channels=1,
            deep_supervision=True,
            input_size_2d=INPUT_SIZE_2D,
            input_size_3d=input_size_3d,
        ).to(device)
        model.eval()

        # load ckpt
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"], strict=True)
        else:
            model.load_state_dict(state, strict=True)

        # 每个 ckpt 独立子目录，避免覆盖
        ckpt_tag = ckpt.stem  # epoch_10 / best / last ...
        ckpt_pred_dir = preds_root / ckpt_tag
        if args.save_preds:
            ckpt_pred_dir.mkdir(parents=True, exist_ok=True)

        psnrs, ssims, maes, nrmses = [], [], [], []

        if spatial_dim == 2:
            # ========================= 2D：逻辑保持原样 =========================
            for p_in, p_gt in tqdm(list(zip(files_in, files_gt)), leave=False, ncols=110):
                x2d = _read_2d_tif_float32(p_in) / IN_MAX
                x = torch.from_numpy(x2d)[None, None].to(device)  # (1,1,H,W)

                out = model(x)
                pred = forward_get_highres(out).squeeze().detach().cpu().numpy()  # (H,W)
                pred = np.clip(pred, 0, 1) * OUT_MAX

                gt = _read_2d_tif_float32(p_gt)

                ps, ss, ma, nr = metrics_2d(pred, gt)
                psnrs.append(ps); ssims.append(ss); maes.append(ma); nrmses.append(nr)

                if args.save_preds:
                    _save_pred_tif(ckpt_pred_dir / p_in.name, pred, args.save_uint16)

        else:
            # ========================= 3D：按窗口堆叠 =========================
            depth = args.stack_depth
            stride = args.stack_stride
            n = len(files_in)
            if n < depth:
                raise RuntimeError(f"Not enough test images for stack_depth={depth}: N={n}")

            # 对每个窗口做一次 volume 推理（(1,1,D,H,W)）
            for start in tqdm(range(0, n - depth + 1, stride), leave=False, ncols=110):
                vol_in = make_stack_by_files(files_in, start, depth) / IN_MAX    # (D,H,W)
                vol_gt = make_stack_by_files(files_gt, start, depth)             # (D,H,W)

                x = torch.from_numpy(vol_in)[None, None].to(device)  # (1,1,D,H,W)

                out = model(x)
                pred = forward_get_highres(out).squeeze().detach().cpu().numpy()

                # 期望输出是 (D,H,W)。若输出是 (H,W)，说明网络/输出头与你的3D目标不一致
                if pred.ndim == 2:
                    raise RuntimeError(
                        f"3D net returned 2D output shape={pred.shape}. "
                        f"Expected (D,H,W). Check net/heads."
                    )
                pred = np.clip(pred, 0, 1) * OUT_MAX  # (D,H,W)

                ps, ss, ma, nr = metrics_3d(pred, vol_gt)
                psnrs.append(ps); ssims.append(ss); maes.append(ma); nrmses.append(nr)

                if args.save_preds:
                    # 用窗口中心帧命名（更直观）
                    center_name = files_in[start + depth // 2].name
                    _save_pred_tif(ckpt_pred_dir / center_name, pred, args.save_uint16)

        results.append([
            args.net,
            ckpt.name,
            float(np.mean(psnrs)) if psnrs else 0.0,
            float(np.mean(ssims)) if ssims else 0.0,
            float(np.mean(maes)) if maes else 0.0,
            float(np.mean(nrmses)) if nrmses else 0.0,
        ])

    # 保存 CSV
    out_csv = net_dir / "metrics_sweep.csv"
    np.savetxt(
        out_csv,
        results,
        delimiter=",",
        header="net,checkpoint,psnr,ssim,mae,nrmse",
        comments="",
        fmt="%s",
    )

    print(f"\nSaved metrics to: {out_csv}")
    if args.save_preds:
        print(f"Saved preds to: {preds_root}")


if __name__ == "__main__":
    main()
