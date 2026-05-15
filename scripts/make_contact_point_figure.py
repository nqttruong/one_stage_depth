"""Fig. 4 — Contact-point representation vs. 3D box-center projection.

For each sample KITTI image, draws GT 2D bbox plus two projected reference
points:
  • Red ✕  — 3D box-center projection: (X, Y - h/2, Z) → image plane
  • Green ● — Contact point: KITTI loc = (X, Y, Z) (bottom of 3D box,
              on the ground plane) → image plane

The contact point is on the ground plane (a known reference frame),
whereas the box-center projection has no such anchor, making it a
poor target for monocular regression.

Usage:
    python scripts/make_contact_point_figure.py \\
        --kitti-root /data/kitti/training \\
        --ids 004839 000301 005282 \\
        --out figs/contact_point.pdf
"""

import argparse
import math
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ogcde.dataset import parse_kitti_label, parse_kitti_calib

plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     9,
    "pdf.fonttype":  42,
    "ps.fonttype":   42,
    "axes.linewidth": 0.6,
})

KEEP_TYPES = {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist"}


def project_3d_to_2d(pt3d, P2):
    """(X,Y,Z) in camera coords → (u, v) pixel."""
    h = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0])
    p = P2 @ h
    return p[0] / p[2], p[1] / p[2]


def load_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--ids",        nargs="+",
                    default=["004839", "000301", "005282"])
    ap.add_argument("--out",        default="figs/contact_point.pdf")
    ap.add_argument("--max-objs",   type=int, default=3,
                    help="Max objects to annotate per image")
    return ap.parse_args()


def annotate_image(ax, img_rgb, objects, P2, max_objs=3):
    """Draw bboxes + contact point + 3D-center projection for top-N objects."""
    H, W = img_rgb.shape[:2]
    ax.imshow(img_rgb, interpolation="lanczos")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)         # image coords: y=0 at top
    ax.axis("off")

    objs = [o for o in objects if o["type"] in KEEP_TYPES]
    if not objs:
        return

    # filter: keep only objects whose 2D bbox is inside the image
    def in_image(o):
        l, t, r, b = o["bbox"]
        return 0 <= l < W and 0 <= t < H and r > 0 and b > 0
    objs = [o for o in objs if in_image(o)]

    def area(o):
        l, t, r, b = o["bbox"]
        return (r - l) * (b - t)
    objs.sort(key=area, reverse=True)
    objs = objs[:max_objs]

    for obj in objs:
        # 2D bbox (KITTI label space — same as image space)
        l, t, r, b = obj["bbox"]
        ax.add_patch(mpatches.Rectangle(
            (l, t), r - l, b - t,
            fill=False, edgecolor="white", linewidth=1.2, alpha=0.9))

        h, w, ll = obj["dims"]  # height, width, length
        x, y, z  = obj["loc"]   # bottom-center in cam coords

        # contact point = bottom-center (ground plane projection)
        cu, cv = project_3d_to_2d((x, y, z), P2)
        # 3D box center = halfway up the box (cam coords Y goes downward, so up = -Y)
        bcu, bcv = project_3d_to_2d((x, y - h / 2, z), P2)

        # draw line connecting them
        ax.plot([cu, bcu], [cv, bcv], "-", color="#FFFFFF",
                lw=0.6, alpha=0.6, zorder=4)

        # 3D-center projection — red ✕
        ax.plot(bcu, bcv, "x", color="#D55E00", markersize=8,
                markeredgewidth=1.8, zorder=6)

        # contact point — green ●
        ax.plot(cu, cv, "o", color="white", markersize=8, zorder=6)
        ax.plot(cu, cv, "o", color="#00B050", markersize=5.5, zorder=7)

        # distance label
        dist = math.sqrt(x*x + y*y + z*z)
        ax.text(l + 2, max(0, t - 3), f"{obj['type']}  {dist:.1f}m",
                color="white", fontsize=7, fontweight="bold",
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.12",
                          facecolor="#444444", alpha=0.85, edgecolor="none"),
                zorder=8)


def main():
    args = load_args()
    n = len(args.ids)
    fig, axes = plt.subplots(n, 1, figsize=(7.16, 1.95 * n),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, img_id in zip(axes, args.ids):
        img_path  = os.path.join(args.kitti_root, "image_2", f"{img_id}.png")
        calib     = parse_kitti_calib(
            os.path.join(args.kitti_root, "calib", f"{img_id}.txt"))
        P2        = calib["P2"]
        objects   = parse_kitti_label(
            os.path.join(args.kitti_root, "label_2", f"{img_id}.txt"))

        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        annotate_image(ax, rgb, objects, P2, max_objs=args.max_objs)
        ax.set_title(f"Sample {img_id}", fontsize=8.5, pad=3, loc="left")

    # Legend below figure
    legend_elems = [
        plt.Line2D([0], [0], marker="x", color="w", label="3D box center projection",
                   markerfacecolor="#D55E00", markeredgecolor="#D55E00",
                   markersize=8, markeredgewidth=1.8),
        plt.Line2D([0], [0], marker="o", color="w", label="Contact point (ground plane)",
                   markerfacecolor="#00B050", markeredgecolor="white",
                   markersize=8),
        plt.Line2D([0], [0], color="white", lw=1.2, label="GT 2D bbox"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=2.0, columnspacing=2.0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Lower DPI to keep file size under 2 MB; KITTI images are 1242×375
    # so 200 DPI is plenty for 7-inch column width.
    fig.savefig(args.out, dpi=200, bbox_inches="tight", format="pdf")
    png = args.out.replace(".pdf", ".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out}  ({os.path.getsize(args.out)/1e6:.2f} MB)")
    print(f"Preview → {png}")


if __name__ == "__main__":
    main()
