"""Chạy OGCDE trên một tập ảnh, vẽ bbox + khoảng cách + kích thước, và đánh giá.

Hai chế độ:

  1. Tập ảnh tùy ý (không có GT):
        python eval_vis.py --ckpt runs/ogcde_v1/best.pt --source /path/to/images/

  2. KITTI split (có GT → vẽ GT box + in metrics cuối):
        python eval_vis.py --ckpt runs/ogcde_v1/best.pt \\
            --kitti-root /data/kitti/training \\
            --split splits/kitti_val.txt

Ảnh visualized được lưu vào --out-dir (mặc định runs/vis/).
Mỗi ảnh annotated được lưu với tên gốc + hậu tố _vis.jpg.

Thông tin hiển thị trên mỗi box:
    [class]  [distance]m  [W]x[H]px
    Ví dụ:  Car  14.3m  132x74px

Legend màu:
    Car        → xanh lá   (0, 200, 0)
    Pedestrian → xanh dương (200, 80, 0)   [BGR]
    Cyclist    → cam        (0, 140, 255)   [BGR]
    GT box     → vàng nhạt  (0, 220, 220)  [BGR] (chỉ ở KITTI mode)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ogcde.model import OGCDENet
from ogcde.utils import decode_predictions, letterbox, unletterbox_boxes, unletterbox_points
from ogcde.metrics import OGCDEEvaluator
from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde


# ─────────────────────────── constants ──────────────────────────────────────

CLASSES = ["Car", "Pedestrian", "Cyclist"]

# BGR colors per class
CLASS_COLORS = {
    0: (0,   200,   0),    # Car       → green
    1: (200,  80,   0),    # Pedestrian→ blue
    2: (0,   140, 255),    # Cyclist   → orange
}

GT_COLOR   = (0, 220, 220)   # GT box:  yellow
CP_RADIUS  = 5               # contact point dot radius (px)


# ─────────────────────────── drawing helpers ────────────────────────────────

def _put_label(img: np.ndarray, text: str, x1: int, y1: int,
               color: tuple, font_scale: float = 0.48,
               thickness: int = 1) -> None:
    """Draw text with a dark background rect for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    ty = max(y1 - 4, th + 4)
    # background
    cv2.rectangle(img,
                  (x1, ty - th - baseline - 2),
                  (x1 + tw + 2, ty + baseline - 2),
                  (20, 20, 20), cv2.FILLED)
    cv2.putText(img, text, (x1 + 1, ty - 1),
                font, font_scale, color, thickness, cv2.LINE_AA)


def draw_dashed_rect(img: np.ndarray, x1: int, y1: int,
                     x2: int, y2: int, color: tuple,
                     dash: int = 8, gap: int = 5,
                     thickness: int = 1) -> None:
    """Draw a dashed rectangle (GT box style)."""
    pts = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in pts:
        total = int(np.hypot(ex - sx, ey - sy))
        if total == 0:
            continue
        dx = (ex - sx) / total
        dy = (ey - sy) / total
        pos = 0
        drawing = True
        while pos < total:
            seg = dash if drawing else gap
            end = min(pos + seg, total)
            if drawing:
                p1 = (int(sx + dx * pos),  int(sy + dy * pos))
                p2 = (int(sx + dx * end),  int(sy + dy * end))
                cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
            pos = end
            drawing = not drawing


def draw_predictions(img: np.ndarray, dets: dict) -> np.ndarray:
    """Vẽ predicted boxes, labels (class + distance + W×H), và contact points."""
    vis = img.copy()
    boxes    = dets["boxes"]      # (N,4) xywh
    scores   = dets["scores"]     # (N,)
    classes  = dets["classes"]    # (N,)
    dists    = dets["distance"]   # (N,)
    contacts = dets["contact"]    # (N,2)

    for i in range(len(boxes)):
        cx, cy, w, h = boxes[i]
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cid   = int(classes[i])
        color = CLASS_COLORS.get(cid, (180, 180, 180))
        dist  = float(dists[i])
        w_px  = max(1, int(w))
        h_px  = max(1, int(h))

        # bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # label: "Car 14.3m 132x74px"
        label = f"{CLASSES[cid]}  {dist:.1f}m  {w_px}x{h_px}px"
        _put_label(vis, label, x1, y1, color)

        # contact point
        u, v = int(contacts[i][0]), int(contacts[i][1])
        cv2.circle(vis, (u, v), CP_RADIUS,     (0, 0, 220), -1)   # filled red
        cv2.circle(vis, (u, v), CP_RADIUS + 1, (255, 255, 255), 1) # white ring

    return vis


def draw_gt_boxes(img: np.ndarray, gt: dict) -> np.ndarray:
    """Vẽ GT boxes (dashed yellow) và GT contact points (cyan)."""
    vis = img
    for i in range(len(gt["boxes"])):
        cx, cy, w, h = gt["boxes"][i]
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        draw_dashed_rect(vis, x1, y1, x2, y2, GT_COLOR, thickness=1)
        # GT contact
        if len(gt["contact"]) > i:
            u, v = int(gt["contact"][i][0]), int(gt["contact"][i][1])
            cv2.circle(vis, (u, v), CP_RADIUS - 1, GT_COLOR, -1)

    return vis


def make_summary_overlay(img: np.ndarray, n_pred: int,
                         n_gt: int | None = None) -> np.ndarray:
    """Góc trên-phải: số lượng dự đoán (và GT nếu có)."""
    vis = img
    H, W = vis.shape[:2]
    if n_gt is not None:
        text = f"pred:{n_pred}  gt:{n_gt}"
    else:
        text = f"pred:{n_pred}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.45, 1
    (tw, tH), base = cv2.getTextSize(text, font, fs, th)
    x = W - tw - 8
    y = tH + 6
    cv2.rectangle(vis, (x - 3, 2), (W - 2, y + base + 2), (20, 20, 20), cv2.FILLED)
    cv2.putText(vis, text, (x, y), font, fs, (220, 220, 220), th, cv2.LINE_AA)
    return vis


# ─────────────────────────── image sources ──────────────────────────────────

def iter_image_folder(folder: str):
    """Yield (bgr_img, stem) từ thư mục ảnh."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in exts
    )
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong: {folder}")
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[skip] không đọc được: {p}")
            continue
        yield img, p.stem


# ─────────────────────────── inference helper ───────────────────────────────

@torch.no_grad()
def infer_single(model: OGCDENet, bgr_img: np.ndarray,
                 img_size: int, obj_thr: float, iou_thr: float,
                 device: torch.device, nc: int):
    """Chạy inference một ảnh. Trả về dets đã unletterbox."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    lb, ratio, pad = letterbox(rgb, img_size)
    t = torch.from_numpy(lb.transpose(2, 0, 1)).contiguous().float() / 255.0
    t = t.unsqueeze(0).to(device)

    preds = model(t)
    dets  = decode_predictions(preds, nc, obj_thr=obj_thr, iou_thr=iou_thr)[0]

    boxes    = unletterbox_boxes(dets["boxes"].cpu().numpy(),    ratio, pad)
    contacts = unletterbox_points(dets["contact"].cpu().numpy(), ratio, pad)
    return {
        "boxes":    boxes,
        "scores":   dets["scores"].cpu().numpy(),
        "classes":  dets["classes"].cpu().numpy(),
        "depth":    dets["depth"].cpu().numpy(),
        "distance": dets["distance"].cpu().numpy(),
        "contact":  contacts,
    }


# ─────────────────────────── mode 1: plain folder ───────────────────────────

def run_folder(args, model, device):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = saved = 0
    for bgr_img, stem in iter_image_folder(args.source):
        dets = infer_single(model, bgr_img, args.img_size,
                            args.obj_thr, args.iou_thr, device, args.num_classes)
        vis = draw_predictions(bgr_img, dets)
        vis = make_summary_overlay(vis, n_pred=len(dets["boxes"]))

        out_path = out_dir / f"{stem}_vis.jpg"
        cv2.imwrite(str(out_path), vis)
        total += 1
        saved += 1
        if args.verbose:
            print(f"  [{total:04d}] {stem}  det={len(dets['boxes'])}"
                  f"  → {out_path.name}")

    print(f"\n[done] {saved}/{total} ảnh lưu tại: {out_dir}/")


# ─────────────────────────── mode 2: KITTI split ────────────────────────────

def run_kitti(args, model, device):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = KITTIOGCDEDataset(
        args.kitti_root, args.split,
        img_size=args.img_size, augment=False,
    )
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=args.workers, collate_fn=collate_ogcde,
    )

    evaluator = OGCDEEvaluator(iou_thr=args.iou_thr, class_agnostic=False)
    total = 0

    for imgs_t, targets, meta in loader:
        m = meta[0]
        img_id = m["image_id"]

        # Đọc ảnh gốc từ disk để visualize ở kích thước thực
        img_path = os.path.join(args.kitti_root, "image_2", f"{img_id}.png")
        bgr_img  = cv2.imread(img_path)
        if bgr_img is None:
            print(f"[skip] {img_id}: không đọc được ảnh")
            continue

        dets = infer_single(model, bgr_img, args.img_size,
                            args.obj_thr, args.iou_thr, device, args.num_classes)

        # Lấy GT của ảnh này (đã letterbox → unletterbox về original coords)
        bmask = (targets["batch_idx"].numpy() == 0)
        gt = {
            "boxes":   unletterbox_boxes(
                            targets["boxes"].numpy()[bmask], m["ratio"], m["pad"]),
            "labels":  targets["labels"].numpy()[bmask],
            "dist":    targets["dist"].numpy()[bmask],
            "depth":   targets["depth"].numpy()[bmask],
            "contact": unletterbox_points(
                            targets["contact"].numpy()[bmask], m["ratio"], m["pad"]),
        }

        # Vẽ GT trước (dưới), pred sau (trên)
        vis = draw_gt_boxes(bgr_img.copy(), gt)
        vis = draw_predictions(vis, dets)
        vis = make_summary_overlay(vis, n_pred=len(dets["boxes"]),
                                   n_gt=len(gt["boxes"]))

        out_path = out_dir / f"{img_id}_vis.jpg"
        cv2.imwrite(str(out_path), vis)
        total += 1

        if args.verbose:
            print(f"  [{total:04d}] {img_id}  det={len(dets['boxes'])}"
                  f"  gt={len(gt['boxes'])}  → {out_path.name}")

        # Cập nhật evaluator
        pred_eval = {
            "boxes":    dets["boxes"],
            "scores":   dets["scores"],
            "classes":  dets["classes"],
            "depth":    dets["depth"],
            "distance": dets["distance"],
            "contact":  dets["contact"],
        }
        evaluator.update(pred_eval, gt)

    # ── In metrics tổng hợp ──
    print(f"\n[done] {total} ảnh visualized → {out_dir}/")

    result = evaluator.compute()
    print("\n" + "=" * 58)
    print(f"  Matched pairs (IoU>{args.iou_thr}): {result['n_matched']}")
    print("\n  [Distance — sqrt(x²+y²+z²)]")
    dm = result["distance"]
    print(f"    AbsRel   = {dm['AbsRel']:.4f}")
    print(f"    RMSE     = {dm['RMSE']:.4f} m")
    print(f"    RMSE_log = {dm['RMSE_log']:.4f}")
    print(f"    δ₁<1.25  = {dm['delta1']*100:.2f}%")
    print(f"    δ₂<1.25² = {dm['delta2']*100:.2f}%")
    print(f"    δ₃<1.25³ = {dm['delta3']*100:.2f}%")
    print("\n  [Depth — Z-axis]")
    dz = result["depth"]
    print(f"    AbsRel   = {dz['AbsRel']:.4f}")
    print(f"    RMSE     = {dz['RMSE']:.4f} m")
    print(f"    δ₁<1.25  = {dz['delta1']*100:.2f}%")
    print(f"\n  DE  (mean dist error)      = {result['DE']:.4f} m")
    print(f"  CPE (contact point error)  = {result['CPE_px']:.4f} px")
    print("=" * 58)

    return result


# ─────────────────────────── CLI ────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Visualize OGCDE detections trên tập ảnh, "
                    "tùy chọn đánh giá với KITTI GT.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--ckpt",       required=True,
                    help="Đường dẫn checkpoint (.pt)")
    ap.add_argument("--source",     default=None,
                    help="Thư mục ảnh (mode không có GT)")
    ap.add_argument("--kitti-root", default=None,
                    help="KITTI root (training/); cần kèm --split")
    ap.add_argument("--split",      default=None,
                    help="File split KITTI (kitti_val.txt)")
    ap.add_argument("--out-dir",    default="runs/vis")
    ap.add_argument("--img-size",   type=int,   default=640)
    ap.add_argument("--obj-thr",    type=float, default=0.25)
    ap.add_argument("--iou-thr",    type=float, default=0.5)
    ap.add_argument("--num-classes",type=int,   default=3)
    ap.add_argument("--workers",    type=int,   default=4)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verbose",    action="store_true",
                    help="In tên file từng ảnh đã xử lý")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.kitti_root and args.split:
        mode = "kitti"
    elif args.source:
        mode = "folder"
    else:
        raise ValueError(
            "Cần chỉ định --source (thư mục ảnh) "
            "hoặc --kitti-root + --split (KITTI mode)."
        )

    device = torch.device(args.device)
    model  = OGCDENet(nc=args.num_classes).to(device)
    ckpt   = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.ckpt}  (epoch {ckpt.get('epoch', '?')},"
          f" val_loss={ckpt.get('val_loss', float('nan')):.3f})")
    print(f"Mode: {mode}  |  out: {args.out_dir}\n")

    if mode == "kitti":
        run_kitti(args, model, device)
    else:
        run_folder(args, model, device)


if __name__ == "__main__":
    main()
