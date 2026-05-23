"""Fig. — Contact point vs. 2D box center on occluded objects.

For occluded objects the visible 2D bounding box shrinks or shifts, pulling
the box centre away from the true ground-plane anchor. The contact point
(projected 3D bottom-centre) stays tied to the ground plane regardless of
how much of the object is visible.

Selects KITTI val images that contain objects with occlusion level >= 1 and
the largest pixel distance between box centre and contact point (most visible
difference). Draws for each object:
  • White rectangle — GT 2D bounding box (visible extent)
  • Red   ✕         — 2D box centre  = ((x1+x2)/2, (y1+y2)/2)
  • Green ●         — Contact point  = projected 3D bottom-centre

Usage:
    python scripts/make_cp_vs_center_figure.py \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt \\
        --out figs/cp_vs_center.pdf
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def parse_label_with_occ(path):
    """Return list of dicts including occlusion level."""
    objs = []
    with open(path) as f:
        for raw_idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 15 or parts[0] == "DontCare":
                continue
            objs.append({
                "type":  parts[0],
                "trunc": float(parts[1]),
                "occ":   int(parts[2]),
                "bbox":  [float(x) for x in parts[4:8]],   # x1,y1,x2,y2
                "dims":  [float(x) for x in parts[8:11]],   # h,w,l
                "loc":   [float(x) for x in parts[11:14]],  # X,Y,Z
                "idx":   raw_idx,
            })
    return objs


def project(pt3d, P2):
    h = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0])
    p = P2 @ h
    return p[0] / p[2], p[1] / p[2]


def scan_for_candidates(kitti_root, img_ids, min_occ=1, top_k=20,
                        max_trunc=0.4, img_w=1242, img_h=375):
    """Find (img_id, obj) pairs with large cp-vs-center pixel gap.

    Filters:
    - occlusion >= min_occ
    - truncation <= max_trunc  (exclude border-cropped objects)
    - contact point projects INSIDE the image (gap is real, not off-screen)
    - box centre also inside image
    """
    candidates = []
    for img_id in img_ids:
        label_path = os.path.join(kitti_root, "label_2", f"{img_id}.txt")
        calib_path = os.path.join(kitti_root, "calib", f"{img_id}.txt")
        if not os.path.exists(label_path):
            continue
        objs  = parse_label_with_occ(label_path)
        calib = parse_kitti_calib(calib_path)
        P2    = calib["P2"]

        for obj in objs:
            if obj["type"] not in KEEP_TYPES:
                continue
            if obj["occ"] < min_occ:
                continue
            if obj["trunc"] > max_trunc:       # skip heavily truncated
                continue

            x1, y1, x2, y2 = obj["bbox"]
            box_cx = (x1 + x2) / 2
            box_cy = (y1 + y2) / 2

            X, Y, Z = obj["loc"]
            if Z < 1.0:
                continue
            cp_u, cp_v = project((X, Y, Z), P2)

            # both points must be inside image
            margin = 20
            if not (margin < cp_u < img_w - margin and margin < cp_v < img_h - margin):
                continue
            if not (margin < box_cx < img_w - margin and margin < box_cy < img_h - margin):
                continue

            gap = math.sqrt((box_cx - cp_u)**2 + (box_cy - cp_v)**2)
            if gap < 10:                        # too small to illustrate
                continue
            candidates.append((gap, img_id, obj, box_cx, box_cy, cp_u, cp_v))

    candidates.sort(reverse=True)
    return candidates[:top_k]


def annotate_ax(ax, img_rgb, objects_info, P2):
    H, W = img_rgb.shape[:2]
    ax.imshow(img_rgb, interpolation="lanczos")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    for obj, box_cx, box_cy, cp_u, cp_v in objects_info:
        x1, y1, x2, y2 = obj["bbox"]

        # GT 2D bbox
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="white", linewidth=1.2, alpha=0.9))

        # line connecting two points
        ax.plot([box_cx, cp_u], [box_cy, cp_v],
                "-", color="white", lw=0.8, alpha=0.6, zorder=4)

        # 2D box centre — red ✕
        ax.plot(box_cx, box_cy, "x", color="#D55E00",
                markersize=9, markeredgewidth=2.0, zorder=6)

        # contact point — green ●
        ax.plot(cp_u, cp_v, "o", color="white",  markersize=9, zorder=6)
        ax.plot(cp_u, cp_v, "o", color="#00B050", markersize=6,  zorder=7)

        # pixel gap label
        gap = math.sqrt((box_cx - cp_u)**2 + (box_cy - cp_v)**2)
        X, Y, Z = obj["loc"]
        dist = math.sqrt(X*X + Y*Y + Z*Z)
        occ_str = ["vis", "part", "larg"][min(obj["occ"], 2)]
        ax.text(x1 + 2, max(0, y1 - 3),
                f"{obj['type']}  {dist:.1f}m  occ={occ_str}  gap={gap:.0f}px",
                color="white", fontsize=6.5, fontweight="bold",
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.12",
                          facecolor="#333333", alpha=0.85, edgecolor="none"),
                zorder=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--split",      default="splits/distformer_val.txt")
    ap.add_argument("--out",        default="figs/cp_vs_center.pdf")
    ap.add_argument("--n-images",   type=int, default=3,
                    help="Number of images to show")
    ap.add_argument("--min-occ",    type=int, default=1,
                    help="Minimum KITTI occlusion level (1=partly, 2=largely)")
    ap.add_argument("--objs-per-img", type=int, default=3,
                    help="Max objects to annotate per image")
    args = ap.parse_args()

    with open(args.split) as f:
        img_ids = [ln.strip() for ln in f if ln.strip()]

    print(f"Scanning {len(img_ids)} images for occ>={args.min_occ} objects…")
    candidates = scan_for_candidates(
        args.kitti_root, img_ids, min_occ=args.min_occ, top_k=200)

    # group by image, pick top-N images with most / largest gaps
    from collections import defaultdict
    img_objs = defaultdict(list)
    for gap, img_id, obj, bx, by, cu, cv in candidates:
        img_objs[img_id].append((obj, bx, by, cu, cv))

    # rank images by sum of top-3 gaps
    def img_score(img_id):
        items = img_objs[img_id]
        gaps  = [math.sqrt((bx-cu)**2 + (by-cv)**2)
                 for _, bx, by, cu, cv in items]
        return sum(sorted(gaps, reverse=True)[:3])

    selected = sorted(img_objs.keys(), key=img_score, reverse=True)[:args.n_images]

    print(f"Selected images: {selected}")

    fig, axes = plt.subplots(args.n_images, 1,
                              figsize=(7.16, 2.1 * args.n_images),
                              constrained_layout=True)
    if args.n_images == 1:
        axes = [axes]

    for ax, img_id in zip(axes, selected):
        img_path = os.path.join(args.kitti_root, "image_2", f"{img_id}.png")
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        calib = parse_kitti_calib(
            os.path.join(args.kitti_root, "calib", f"{img_id}.txt"))
        P2    = calib["P2"]

        items = img_objs[img_id]
        items = sorted(items,
                        key=lambda t: math.sqrt((t[1]-t[3])**2 + (t[2]-t[4])**2),
                        reverse=True)[:args.objs_per_img]

        annotate_ax(ax, rgb, items, P2)
        ax.set_title(f"Sample {img_id}", fontsize=8, pad=2, loc="left")

    # legend
    legend_elems = [
        plt.Line2D([0], [0], marker="x", color="w",
                   label="2D box centre  (shifts with occlusion)",
                   markerfacecolor="#D55E00", markeredgecolor="#D55E00",
                   markersize=8, markeredgewidth=2.0),
        plt.Line2D([0], [0], marker="o", color="w",
                   label="Contact point  (ground-plane anchor)",
                   markerfacecolor="#00B050", markeredgecolor="white",
                   markersize=8),
        plt.Line2D([0], [0], color="white", lw=1.2,
                   label="GT 2D bbox"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.8, columnspacing=1.8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", format="pdf")
    png = args.out.replace(".pdf", ".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out}  ({os.path.getsize(args.out)/1e6:.2f} MB)")
    print(f"Preview → {png}")


if __name__ == "__main__":
    main()
