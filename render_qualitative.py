"""Render publication-quality qualitative figures for OGCDE paper.

5 scenarios (pass via --images):
  easy        000848  — 4 cars, 16-32m, straight road
  far_range   004839  — cars from 11m to 80m
  oblique     000301  — cars at large lateral angle
  crowded     004825  — 1 car + 9 pedestrians
  failure     002615  — parking lot, heavy occlusion

Usage:
    python render_qualitative.py \\
        --ckpt runs/ogcde_lidar_p2/best.pt \\
        --kitti-root /data/kitti/training \\
        --out-dir figures/qualitative \\
        --images 000848 004839 000301 004825 002615 \\
        --labels easy far_range oblique crowded failure
"""

from __future__ import annotations

import argparse
import math
import os

import cv2
import numpy as np
import torch

from ogcde.model import OGCDENet
from ogcde.dataset import parse_kitti_label, parse_kitti_calib, letterbox
from ogcde.utils import decode_predictions, unletterbox_boxes, unletterbox_points

# ─── color scheme (Wong color-blind safe palette, BGR) ────────────────────
COLORS = {
    0: (178, 114,   0),   # Car        — blue   #0072B2
    1: (  0, 159, 230),   # Pedestrian — orange #E69F00
    2: (115, 158,   0),   # Cyclist    — green  #009E73
}
CLASS_NAMES = {0: "Car", 1: "Ped", 2: "Cyc"}
GT_COLOR    = (255, 255, 255)   # white for GT boxes

CLASS_MAP = {"Car": 0, "Van": 0, "Truck": 0,
             "Pedestrian": 1, "Person_sitting": 1, "Cyclist": 2}


# ─── helpers ──────────────────────────────────────────────────────────────

def load_image(kitti_root, img_id):
    path = os.path.join(kitti_root, "image_2", f"{img_id}.png")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def draw_label_bg(img, x, y, text, color, font=cv2.FONT_HERSHEY_DUPLEX,
                  scale=0.55, thick=1):
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    pad = 3
    cv2.rectangle(img, (x - pad, y - th - pad - bl),
                  (x + tw + pad, y + pad), color, -1)
    cv2.putText(img, text, (x, y - bl),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def render_one(img_orig, pred_boxes_xywh, pred_scores, pred_classes,
               pred_depths, pred_scales, pred_contact,
               gt_objects=None, score_thr=0.25, show_gt=True):
    """
    Returns rendered BGR image.
    pred_* are numpy arrays in original image coordinates.
    """
    img = img_orig.copy()
    H, W = img.shape[:2]

    # ── GT overlay (white dashed boxes) ──────────────────────────────────
    if show_gt and gt_objects:
        for obj in gt_objects:
            if obj["type"] not in CLASS_MAP:
                continue
            l, t, r, b = [int(v) for v in obj["bbox"]]
            # dashed rectangle (4 corner segments)
            seg = 8
            for x in range(l, r, seg * 2):
                cv2.line(img, (x, t), (min(x+seg, r), t), GT_COLOR, 1)
                cv2.line(img, (x, b), (min(x+seg, r), b), GT_COLOR, 1)
            for y in range(t, b, seg * 2):
                cv2.line(img, (l, y), (l, min(y+seg, b)), GT_COLOR, 1)
                cv2.line(img, (r, y), (r, min(y+seg, b)), GT_COLOR, 1)

    # ── predictions ──────────────────────────────────────────────────────
    for i in range(len(pred_scores)):
        if pred_scores[i] < score_thr:
            continue

        c    = int(pred_classes[i])
        col  = COLORS.get(c, (200, 200, 200))
        name = CLASS_NAMES.get(c, "?")

        cx, cy, w, h = pred_boxes_xywh[i]
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(W - 1, int(cx + w / 2))
        y2 = min(H - 1, int(cy + h / 2))

        depth    = pred_depths[i]
        scale    = pred_scales[i]
        distance = depth * scale

        # bounding box (2px solid)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

        # contact point
        cpx, cpy = int(pred_contact[i, 0]), int(pred_contact[i, 1])
        cpx = max(0, min(W - 1, cpx))
        cpy = max(0, min(H - 1, cpy))
        cv2.circle(img, (cpx, cpy), 5, col, -1)
        cv2.circle(img, (cpx, cpy), 6, (255, 255, 255), 1)

        # label: "Car  D=23.4m"
        label = f"{name}  D={distance:.1f}m"
        draw_label_bg(img, x1, y1 - 2, label, col)

    return img


# ─── inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model, img_orig, img_size=640, device="cuda",
          conf=0.25, nms_iou=0.45, nc=3):
    img_lb, ratio, (px, py) = letterbox(img_orig, img_size)
    x = torch.from_numpy(
        img_lb.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)

    preds = model(x)
    dets  = decode_predictions(preds, nc, obj_thr=conf, iou_thr=nms_iou)
    det   = dets[0]

    meta = {"ratio": ratio, "pad": (px, py)}

    boxes   = unletterbox_boxes(det["boxes"].cpu().numpy(), ratio, (px, py))
    contact = unletterbox_points(det["contact"].cpu().numpy(), ratio, (px, py))
    depth   = det["depth"].cpu().numpy()
    dist    = det["distance"].cpu().numpy()
    # scale = distance / depth
    scale   = np.where(depth > 0, dist / np.clip(depth, 1e-6, None),
                       np.ones_like(dist))

    return {
        "boxes":   boxes,          # xywh original coords
        "scores":  det["scores"].cpu().numpy(),
        "classes": det["classes"].cpu().numpy(),
        "depth":   depth,
        "scale":   scale,
        "distance": dist,
        "contact": contact,
    }


# ─── main ─────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--out-dir",    default="figures/qualitative")
    ap.add_argument("--images",     nargs="+", required=True,
                    help="Image IDs (6-digit strings)")
    ap.add_argument("--labels",     nargs="+",
                    help="Scenario label for each image (same order as --images)")
    ap.add_argument("--img-size",   type=int,   default=640)
    ap.add_argument("--conf",       type=float, default=0.25)
    ap.add_argument("--nms-iou",    type=float, default=0.45)
    ap.add_argument("--score-thr",  type=float, default=0.25,
                    help="Display threshold for rendering")
    ap.add_argument("--no-gt",      action="store_true",
                    help="Do not render GT boxes")
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    labels = args.labels or args.images

    # load model
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    bs    = ckpt.get("args", {}).get("backbone_size", "n")
    nc    = ckpt.get("args", {}).get("num_classes", 3)
    model = OGCDENet(nc=nc, backbone_size=bs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for img_id, label in zip(args.images, labels):
        print(f"[{label}]  ID={img_id} ...", end=" ", flush=True)

        img_orig = load_image(args.kitti_root, img_id)

        # GT for overlay
        label_path = os.path.join(args.kitti_root, "label_2", f"{img_id}.txt")
        gt_objects = parse_kitti_label(label_path)

        # inference
        pred = infer(model, img_orig, img_size=args.img_size, device=device,
                     conf=args.conf, nms_iou=args.nms_iou, nc=nc)

        # render
        rendered = render_one(
            img_orig,
            pred["boxes"], pred["scores"], pred["classes"],
            pred["depth"], pred["scale"], pred["contact"],
            gt_objects=None if args.no_gt else gt_objects,
            score_thr=args.score_thr,
            show_gt=not args.no_gt,
        )

        out_path = os.path.join(args.out_dir, f"{label}_{img_id}.png")
        cv2.imwrite(out_path, rendered, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"→ {out_path}  ({rendered.shape[1]}×{rendered.shape[0]})")

    print(f"\nDone. All figures saved in: {args.out_dir}/")
    print("Tip: KITTI images are 1242×375. For paper, crop or pad to 16:9 "
          "before including in LaTeX.")


if __name__ == "__main__":
    main()
