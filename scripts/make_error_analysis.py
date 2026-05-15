"""Fig. 6 — Error analysis plots.

(a) Distance error vs. ground-truth distance (binned, median ± IQR)
(b) AbsRel by class and object size (small / medium / large)

Cache matched-pair statistics in figs/error_cache.npz to avoid re-running
inference on every plot tweak.

Usage:
    # First run — runs inference, caches data, plots:
    python scripts/make_error_analysis.py \\
        --ckpt runs/ogcde_lidar_p2/best.pt \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt \\
        --lidar-gt cache/lidar_gt_val.json \\
        --out figs/error_analysis.pdf

    # Subsequent runs — uses cache:
    python scripts/make_error_analysis.py --use-cache --out figs/error_analysis.pdf
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     9,
    "pdf.fonttype":  42,
    "ps.fonttype":   42,
    "axes.linewidth": 0.6,
})

CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
CLASS_COLORS = {0: "#0072B2", 1: "#E69F00", 2: "#009E73"}


def collect_pairs(args):
    """Run inference, collect matched (pred, GT, class, bbox_height_px) per object."""
    import torch
    from torch.utils.data import DataLoader
    from ogcde.model import OGCDENet
    from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde
    from ogcde.utils import decode_predictions, unletterbox_boxes
    from ogcde.metrics import xywh_to_xyxy, bbox_iou_xyxy

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    use_d = ck.get("args", {}).get("no_s_head", False)
    nc    = ck.get("args", {}).get("num_classes", 3)
    bs    = ck.get("args", {}).get("backbone_size", "n")
    model = OGCDENet(nc=nc, backbone_size=bs).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    with open(args.lidar_gt) as f:
        lidar = json.load(f)

    ds = KITTIOGCDEDataset(args.kitti_root, args.split,
                           img_size=640, augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False,
                        num_workers=4, collate_fn=collate_ogcde)

    pred_d, gt_d, cls_arr, bh_px = [], [], [], []

    with torch.no_grad():
        for imgs, targets, meta in loader:
            imgs = imgs.to(device, non_blocking=True)
            dets = decode_predictions(model(imgs), nc, obj_thr=0.25, iou_thr=0.5)
            for b, det in enumerate(dets):
                m  = meta[b]
                bm = targets["batch_idx"].numpy() == b
                pb = unletterbox_boxes(det["boxes"].cpu().numpy(), m["ratio"], m["pad"])
                px = xywh_to_xyxy(pb)
                gb = unletterbox_boxes(targets["boxes"].numpy()[bm], m["ratio"], m["pad"])
                gx = xywh_to_xyxy(gb)
                pd = (det["depth"] if use_d else det["distance"]).cpu().numpy()
                pc = det["classes"].cpu().numpy()
                gl = targets["labels"].numpy()[bm]
                go = targets["obj_idx"].numpy()[bm]
                if not len(px) or not len(gx): continue
                ious = bbox_iou_xyxy(px, gx)
                lr_db = {r["obj_idx"]: r for r in lidar.get(str(m["image_id"]), [])}

                for pi in range(len(px)):
                    row = ious[pi].copy(); row[gl != int(pc[pi])] = -1
                    if row.max() < 0.5: continue
                    gi = int(row.argmax())
                    r  = lr_db.get(int(go[gi]))
                    if not r or r.get("n_points", 0) == 0: continue
                    # bbox height in original-image pixels
                    bh = float(gx[gi, 3] - gx[gi, 1])
                    pred_d.append(float(pd[pi]))
                    gt_d.append(float(r["dist_lidar"]))
                    cls_arr.append(int(pc[pi]))
                    bh_px.append(bh)

    return (np.array(pred_d), np.array(gt_d),
            np.array(cls_arr, dtype=np.int64),
            np.array(bh_px))


# ─── plotting ──────────────────────────────────────────────────────────────

def plot_a_distance_error_by_range(ax, pred, gt, cls):
    """Panel (a): median |error| with IQR band, binned by GT distance."""
    err = np.abs(pred - gt)
    bins = np.array([0, 10, 20, 30, 40, 50, 60, 80])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c in range(3):
        mask_c = cls == c
        if mask_c.sum() < 30: continue
        medians, q25, q75, ns = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = mask_c & (gt >= lo) & (gt < hi)
            n = int(m.sum())
            if n < 5:
                medians.append(np.nan); q25.append(np.nan); q75.append(np.nan); ns.append(0)
                continue
            e = err[m]
            medians.append(float(np.median(e)))
            q25.append(float(np.percentile(e, 25)))
            q75.append(float(np.percentile(e, 75)))
            ns.append(n)
        medians = np.array(medians); q25 = np.array(q25); q75 = np.array(q75)
        col = CLASS_COLORS[c]
        ax.plot(bin_centers, medians, "-o", color=col, lw=1.4, ms=4,
                label=CLASS_NAMES[c], zorder=4)
        ax.fill_between(bin_centers, q25, q75, color=col, alpha=0.18, zorder=2)

    ax.set_xlabel("Ground-truth distance (m)")
    ax.set_ylabel("Absolute error (m, median + IQR)")
    ax.set_title("(a) Distance error vs. range", fontsize=9, pad=3)
    ax.set_xlim(0, 80)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_b_absrel_by_class_size(ax, pred, gt, cls, bh):
    """Panel (b): AbsRel grouped by class × distance range.

    Bbox-height size bins are not comparable across classes because
    pedestrians/cyclists (1.7 m) are mostly ≥40 px even at far range,
    while cars (1.5 m) span all bins. Distance-based bins give a fairer
    per-class comparison.
    """
    absrel = np.abs(pred - gt) / gt
    dist_bins = [(0, 20,  "Near (<20 m)"),
                 (20, 40, "Mid (20–40 m)"),
                 (40, 80, "Far (40–80 m)")]

    width = 0.25
    x     = np.arange(len(dist_bins))

    for ci, c in enumerate(range(3)):
        means, ns = [], []
        for lo, hi, _ in dist_bins:
            m = (cls == c) & (gt >= lo) & (gt < hi)
            if m.sum() < 5:
                means.append(np.nan); ns.append(0); continue
            means.append(float(np.mean(absrel[m]) * 100))
            ns.append(int(m.sum()))
        bars = ax.bar(x + (ci - 1) * width, means, width,
                      color=CLASS_COLORS[c], edgecolor="black", linewidth=0.4,
                      label=CLASS_NAMES[c])
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, 0.3,
                        "—", ha="center", va="bottom",
                        fontsize=7, color="#888888")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in dist_bins], fontsize=8)
    ax.set_ylabel("AbsRel (%)")
    ax.set_title("(b) AbsRel by class × distance range", fontsize=9, pad=3)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_ylim(0, None)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       default="runs/ogcde_lidar_p2/best.pt")
    ap.add_argument("--kitti-root", default=None)
    ap.add_argument("--split",      default="splits/distformer_val.txt")
    ap.add_argument("--lidar-gt",   default="cache/lidar_gt_val.json")
    ap.add_argument("--out",        default="figs/error_analysis.pdf")
    ap.add_argument("--cache",      default="figs/error_cache.npz")
    ap.add_argument("--use-cache",  action="store_true")
    ap.add_argument("--device",     default="cuda")
    args = ap.parse_args()

    if args.use_cache and os.path.exists(args.cache):
        z = np.load(args.cache)
        pred, gt, cls, bh = z["pred"], z["gt"], z["cls"], z["bh"]
        print(f"Loaded cache: {len(pred)} pairs from {args.cache}")
    else:
        if args.kitti_root is None:
            raise SystemExit("--kitti-root required when not using cache")
        pred, gt, cls, bh = collect_pairs(args)
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        np.savez(args.cache, pred=pred, gt=gt, cls=cls, bh=bh)
        print(f"Cached {len(pred)} pairs → {args.cache}")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.16, 3.0),
                                     constrained_layout=True)
    plot_a_distance_error_by_range(ax_a, pred, gt, cls)
    plot_b_absrel_by_class_size(ax_b, pred, gt, cls, bh)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", format="pdf")
    png = args.out.replace(".pdf", ".png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out}  ({os.path.getsize(args.out)/1e6:.2f} MB)")
    print(f"Preview → {png}")


if __name__ == "__main__":
    main()
