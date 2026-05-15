"""Teaser figure (Fig. 1) for OGCDE paper.

Two-panel composition:
  (a) KITTI image with full pipeline overlay: bbox + contact point +
      predicted distance — single forward pass, end-to-end.
  (b) Accuracy vs. Speed comparison vs. DistFormer.

Usage:
    python scripts/make_teaser_figure.py \\
        --ckpt runs/ogcde_lidar_p2/best.pt \\
        --kitti-root /data/kitti/training \\
        --img-id 004839 \\
        --out figs/teaser.pdf
"""

import argparse
import json
import math
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from ogcde.model import OGCDENet
from ogcde.dataset import letterbox
from ogcde.utils import decode_predictions, unletterbox_boxes, unletterbox_points

plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     9,
    "pdf.fonttype":  42,
    "ps.fonttype":   42,
    "axes.linewidth": 0.6,
})

# Wong colorblind-safe (BGR for OpenCV, RGB for matplotlib)
COLORS_MPL = {0: "#0072B2", 1: "#E69F00", 2: "#009E73"}  # Car, Ped, Cyc
CLASS_NAMES = {0: "Car", 1: "Ped", 2: "Cyc"}


@torch.no_grad()
def infer(model, img_bgr, img_size=640, device="cuda",
          conf=0.30, nms_iou=0.45, nc=3):
    img_lb, ratio, (px, py) = letterbox(img_bgr, img_size)
    x = torch.from_numpy(img_lb.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)
    preds = model(x)
    dets  = decode_predictions(preds, nc, obj_thr=conf, iou_thr=nms_iou)[0]
    return {
        "boxes":    unletterbox_boxes(dets["boxes"].cpu().numpy(), ratio, (px, py)),
        "contact":  unletterbox_points(dets["contact"].cpu().numpy(), ratio, (px, py)),
        "scores":   dets["scores"].cpu().numpy(),
        "classes":  dets["classes"].cpu().numpy(),
        "distance": dets["distance"].cpu().numpy(),
    }


def draw_teaser_image(ax, img_rgb, pred, score_thr=0.3):
    ax.imshow(img_rgb, interpolation="lanczos")
    ax.axis("off")
    H, W = img_rgb.shape[:2]

    for i in range(len(pred["scores"])):
        if pred["scores"][i] < score_thr:
            continue
        c    = int(pred["classes"][i])
        col  = COLORS_MPL.get(c, "#FFFFFF")
        cx, cy, w, h = pred["boxes"][i]
        x1, y1 = max(0, cx - w/2), max(0, cy - h/2)
        x2, y2 = min(W-1, cx + w/2), min(H-1, cy + h/2)
        cpx, cpy = pred["contact"][i]
        dist = pred["distance"][i]

        # bbox
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=col, linewidth=1.6))

        # contact point: bright green dot with white outline
        ax.plot(cpx, cpy, "o", color="white", markersize=8, zorder=5)
        ax.plot(cpx, cpy, "o", color="#00FF66", markersize=5.5, zorder=6)

        # distance label
        label = f"{CLASS_NAMES.get(c, '?')} {dist:.1f}m"
        label_y = y1 - 3 if (y1 > 20 and (y2 - y1) >= 28) else y2 + 12
        ax.text(
            x1 + 2, label_y, label,
            color="white", fontsize=7.5, fontweight="bold",
            verticalalignment="bottom" if label_y < y1 else "top",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=col,
                      alpha=0.9, edgecolor="none"),
            zorder=7,
        )


def draw_accuracy_vs_speed(ax):
    """Speed-accuracy scatter: OGCDE highlighted in fast+accurate quadrant.

    Labels are placed via leader lines (annotate with arrow) at fixed
    positions chosen so that no text overlaps with markers or other text.
    """
    # name, FPS, AbsRel%, marker, color, (label_dx, label_dy in axes units)
    methods = [
        ("DisNet",       50,  26.5,  "s", "#888888", ( 0,  +1.2)),
        ("Zhu et al.",   25,  16.1,  "^", "#888888", (+5,  +0.0)),
        ("DistFormer",   28,  10.4,  "D", "#D55E00", (+15, +2.0)),
        ("CenterNet",    32,   8.7,  "v", "#888888", (+15, -1.4)),
        ("PatchNet",     20,   8.1,  "<", "#888888", (-2,  -2.0)),
        ("OGCDE (ours)", 84,   7.54, "*", "#0072B2", ( 0,  +3.0)),
    ]

    # shaded "ideal" region first (so markers sit on top)
    ax.axvspan(60, 100, ymin=0, ymax=0.35, color="#0072B2", alpha=0.08, zorder=1)
    ax.text(62, 4.5, "real-time + accurate", fontsize=6.5,
            color="#0072B2", ha="left", va="bottom",
            style="italic", alpha=0.75)

    for name, fps, absrel, marker, col, (dx, dy) in methods:
        is_ours = "ours" in name
        size  = 320 if is_ours else 110
        edge  = 1.6 if is_ours else 0.6
        ax.scatter(fps, absrel, s=size, marker=marker,
                   facecolor=col, edgecolor="black", linewidth=edge,
                   zorder=5 if is_ours else 3,
                   alpha=1.0 if is_ours else 0.85)

        # leader-line annotation: text at (fps+dx, absrel+dy)
        text_xy   = (fps + dx, absrel + dy)
        ha = "center" if abs(dx) < 3 else ("left" if dx > 0 else "right")
        # only draw arrow if label is far enough from marker
        arrowprops = (
            dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.5)
            if (abs(dx) >= 5 or abs(dy) >= 1.0) and not is_ours
            else None
        )
        ax.annotate(
            name, xy=(fps, absrel), xytext=text_xy,
            fontsize=7.5, ha=ha, va="center",
            fontweight="bold" if is_ours else "normal",
            color="black",
            arrowprops=arrowprops,
            zorder=6,
        )

    ax.set_xlabel("Inference speed (FPS)", fontsize=8.5)
    ax.set_ylabel("AbsRel (%, lower better)", fontsize=8.5)
    ax.set_xlim(10, 105)
    ax.set_ylim(3.5, 30)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       default="runs/ogcde_lidar_p2/best.pt")
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--img-id",     default="004839")
    ap.add_argument("--out",        default="figs/teaser.pdf")
    ap.add_argument("--conf",       type=float, default=0.30)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ── load model ────────────────────────────────────────────────────────
    ck   = torch.load(args.ckpt, map_location=device, weights_only=False)
    bs   = ck.get("args", {}).get("backbone_size", "n")
    nc   = ck.get("args", {}).get("num_classes", 3)
    model = OGCDENet(nc=nc, backbone_size=bs).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # ── load image + infer ────────────────────────────────────────────────
    img_path = os.path.join(args.kitti_root, "image_2", f"{args.img_id}.png")
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    pred = infer(model, img_bgr, device=device, conf=args.conf, nc=nc)

    # ── compose figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.16, 2.6), constrained_layout=True)
    gs  = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    draw_teaser_image(ax_img, img_rgb, pred, score_thr=args.conf)
    ax_img.set_title(
        "Single forward pass → bbox + contact point + distance (m)",
        fontsize=8.5, pad=4,
    )

    draw_accuracy_vs_speed(ax_plot)
    ax_plot.set_title("Accuracy vs. speed", fontsize=8.5, pad=4)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", format="pdf")
    png = args.out.replace(".pdf", ".png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved → {args.out}  ({size_mb:.2f} MB)")
    print(f"Preview → {png}")


if __name__ == "__main__":
    main()
