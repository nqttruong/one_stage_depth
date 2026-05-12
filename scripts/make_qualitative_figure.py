"""Render 2×2 qualitative figure for paper from selected candidates.

Input:  figs/qualitative_candidates.json
Output: figs/qualitative.pdf  (Type-42 fonts, 300 dpi)

Usage:
    python scripts/make_qualitative_figure.py \\
        --candidates figs/qualitative_candidates.json \\
        --kitti-root /data/kitti/training \\
        --out figs/qualitative.pdf
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

# ── paper-quality matplotlib settings ───────────────────────────────────────
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     9,
    "pdf.fonttype":  42,
    "ps.fonttype":   42,
    "axes.linewidth": 0.5,
})

# Colors (BGR→RGB) — colorblind-safe palette
CLASS_COLORS = {
    0: "#00B4D8",   # Car        — cyan
    1: "#FFD60A",   # Pedestrian — yellow
    2: "#FF006E",   # Cyclist    — magenta
}

PANEL_ORDER = ["panel_a", "panel_b", "panel_c", "panel_d"]


def load_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="figs/qualitative_candidates.json")
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--out",        default="figs/qualitative.pdf")
    ap.add_argument("--pad-factor", type=float, default=0.5,
                    help="Padding as fraction of bbox size on each side")
    return ap.parse_args()


def load_image(kitti_root, img_id):
    path = os.path.join(kitti_root, "image_2", f"{img_id}.png")
    bgr  = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_region(img, x1, y1, x2, y2, pad_factor=0.5):
    """Crop image to bbox + padding. Returns (crop_rgb, offset_x, offset_y)."""
    H, W = img.shape[:2]
    w, h = x2 - x1, y2 - y1
    pad = max(w, h) * pad_factor
    cx1 = max(0, int(x1 - pad))
    cy1 = max(0, int(y1 - pad))
    cx2 = min(W, int(x2 + pad))
    cy2 = min(H, int(y2 + pad))
    return img[cy1:cy2, cx1:cx2], cx1, cy1


def draw_panel(ax, img_rgb, sample, pad_factor):
    """Draw one panel on a matplotlib Axes."""
    pred_xyxy    = sample["pred_box_xyxy"]     # [x1,y1,x2,y2] original coords
    gt_xyxy      = sample["gt_box_xyxy"]
    contact_orig = sample["pred_contact"]       # already in original coords
    pred_dist    = sample["pred_dist"]
    gt_dist      = sample["gt_dist"]
    absrel       = sample["absrel"]
    cls_id       = sample["class_id"]
    color        = CLASS_COLORS.get(cls_id, "#FFFFFF")

    x1, y1, x2, y2 = pred_xyxy
    w, h = x2 - x1, y2 - y1

    # crop
    crop, ox, oy = crop_region(img_rgb, x1, y1, x2, y2, pad_factor)
    ax.imshow(crop, interpolation="lanczos")
    ax.axis("off")

    # helpers: shift coords to crop space
    def cx(x): return x - ox
    def cy(y): return y - oy

    # ── GT box (white dashed) ────────────────────────────────────────────
    gx1, gy1, gx2, gy2 = gt_xyxy
    gt_rect = mpatches.FancyBboxPatch(
        (cx(gx1), cy(gy1)), gx2 - gx1, gy2 - gy1,
        boxstyle="square,pad=0", fill=False,
        edgecolor="white", linewidth=1.0, linestyle="--",
    )
    ax.add_patch(gt_rect)

    # ── Predicted bbox ───────────────────────────────────────────────────
    pred_rect = mpatches.FancyBboxPatch(
        (cx(x1), cy(y1)), w, h,
        boxstyle="square,pad=0", fill=False,
        edgecolor=color, linewidth=1.5,
    )
    ax.add_patch(pred_rect)

    # ── Contact point: green dot + white outline ─────────────────────────
    cpx, cpy = cx(contact_orig[0]), cy(contact_orig[1])
    ax.plot(cpx, cpy, "o", color="white", markersize=7, zorder=5)
    ax.plot(cpx, cpy, "o", color="#00FF66", markersize=5, zorder=6)

    # ── Distance text ─────────────────────────────────────────────────────
    # Predicted distance: above bbox top-left (or inside if enough height)
    label_y = cy(y1) - 3 if h >= 40 else cy(y1) + 12
    ax.text(
        cx(x1) + 2, label_y,
        f"{pred_dist:.1f}m",
        color="white", fontsize=8, fontweight="bold",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="black",
                  alpha=0.65, edgecolor="none"),
        zorder=7,
    )

    # GT distance: bottom-right of pred box
    ax.text(
        cx(x2) - 2, cy(y2) + 2,
        f"GT: {gt_dist:.1f}m",
        color="white", fontsize=6.5,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="#333333",
                  alpha=0.60, edgecolor="none"),
        zorder=7,
    )

    # AbsRel badge (top-right of crop for failure panel)
    if absrel > 0.20:
        crop_h, crop_w = crop.shape[:2]
        ax.text(
            crop_w - 4, 4,
            f"Err {absrel*100:.0f}%",
            color="white", fontsize=6.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#CC0000",
                      alpha=0.85, edgecolor="none"),
            zorder=8,
        )


def main():
    args = load_args()

    with open(args.candidates) as f:
        candidates = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5), constrained_layout=True)
    ax_flat   = axes.flatten()

    for i, panel_id in enumerate(PANEL_ORDER):
        panel  = candidates.get(panel_id, {})
        chosen = panel.get("chosen")
        ax     = ax_flat[i]
        title  = panel.get("title", panel_id)

        if chosen is None:
            ax.text(0.5, 0.5, "No candidate found",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=9, pad=3)
            ax.axis("off")
            continue

        img = load_image(args.kitti_root, chosen["img_id"])
        draw_panel(ax, img, chosen, args.pad_factor)
        ax.set_title(title, fontsize=9, pad=3)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", format="pdf")
    # also save PNG for quick preview
    png_path = args.out.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved → {args.out}  ({size_mb:.2f} MB)")
    print(f"Preview → {png_path}")

    if size_mb > 2.0:
        print(f"⚠️  PDF > 2MB ({size_mb:.2f} MB). "
              "Consider reducing image resolution or number of panels.")


if __name__ == "__main__":
    main()
