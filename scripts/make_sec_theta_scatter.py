"""Fig. — Scatter plot: s_gt vs s_pred, coloured by bearing angle θ.

s = sec(θ) is the geometric scale factor that relates Z-depth to euclidean
distance.  The model explicitly regresses s via the s-head, so this scatter
visualises how well the head learns the geometric relationship.

  s_gt   = dist_gt / depth_gt  =  sqrt(X²+Y²+Z²) / Z   (from KITTI loc)
  s_pred = distance_pred / depth_pred  =  exp(s_raw)     (from model output)
  θ      = arctan(|X| / Z)   (horizontal bearing from optical axis, radians)

Expected pattern: tight cloud along y=x diagonal; spread grows at periphery
(large θ) where small errors in X cause larger sec(θ) variation.

Usage:
    python scripts/make_sec_theta_scatter.py \\
        --ckpt runs/ogcde_lidar_p2/best.pt \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt \\
        --out figs/sec_theta_scatter.pdf

    # re-plot from cache (skip inference):
    python scripts/make_sec_theta_scatter.py --use-cache
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     9,
    "pdf.fonttype":  42,
    "ps.fonttype":   42,
    "axes.linewidth": 0.6,
})

CACHE_PATH = "figs/sec_theta_cache.npz"
KEEP_TYPES = {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist"}


# ── data collection ────────────────────────────────────────────────────────

def collect(args):
    import torch
    from torch.utils.data import DataLoader
    from ogcde.model import OGCDENet
    from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde, parse_kitti_label, parse_kitti_calib
    from ogcde.utils import decode_predictions, unletterbox_boxes
    from ogcde.metrics import xywh_to_xyxy, bbox_iou_xyxy

    device = torch.device(args.device)
    ck   = torch.load(args.ckpt, map_location=device, weights_only=False)
    nc   = ck.get("args", {}).get("num_classes", 3)
    bs   = ck.get("args", {}).get("backbone_size", "n")
    model = OGCDENet(nc=nc, backbone_size=bs).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    with open(args.split) as f:
        img_ids = [ln.strip() for ln in f if ln.strip()]

    ds = KITTIOGCDEDataset(args.kitti_root, args.split,
                           img_size=640, augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False,
                        num_workers=4, collate_fn=collate_ogcde)

    s_gt_list, s_pred_list, theta_list = [], [], []

    # build a fast lookup: img_id -> list of KITTI objects (with raw loc)
    label_db = {}
    for img_id in img_ids:
        label_path = os.path.join(args.kitti_root, "label_2", f"{img_id}.txt")
        if os.path.exists(label_path):
            label_db[img_id] = parse_kitti_label(label_path)

    with torch.no_grad():
        for imgs, targets, meta in loader:
            imgs = imgs.to(device, non_blocking=True)
            dets = decode_predictions(model(imgs), nc, obj_thr=0.25, iou_thr=0.5)

            for b, det in enumerate(dets):
                m   = meta[b]
                img_id = str(m["image_id"]).zfill(6)
                bm  = targets["batch_idx"].numpy() == b

                if not len(det["boxes"]):
                    continue

                # GT boxes (xywh in letterbox space) → xyxy in original space
                gb = unletterbox_boxes(
                    targets["boxes"].numpy()[bm], m["ratio"], m["pad"])
                gx = xywh_to_xyxy(gb)
                gl = targets["labels"].numpy()[bm]

                # pred boxes in original space
                pb = unletterbox_boxes(
                    det["boxes"].cpu().numpy(), m["ratio"], m["pad"])
                px = xywh_to_xyxy(pb)

                # s_pred = distance / depth  (both positive, same sign as s_raw)
                dist_pred  = det["distance"].cpu().numpy()
                depth_pred = det["depth"].cpu().numpy()
                eps = 1e-6
                sp = dist_pred / (depth_pred + eps)

                if not len(gx):
                    continue

                ious = bbox_iou_xyxy(px, gx)   # (n_pred, n_gt)

                # full (unfiltered) label list — obj_idx is an index into this
                all_objs = label_db.get(img_id, [])
                obj_idx_arr = targets["obj_idx"].numpy()[bm]

                for pi in range(len(px)):
                    row = ious[pi].copy()
                    row[gl != int(det["classes"][pi].item())] = -1
                    if row.max() < 0.5:
                        continue
                    gi = int(row.argmax())

                    # obj_idx[gi] is the raw index in the full label list
                    oi = int(obj_idx_arr[gi]) if gi < len(obj_idx_arr) else -1
                    if oi < 0 or oi >= len(all_objs):
                        continue
                    X, Y, Z = all_objs[oi]["loc"]
                    if Z < 0.5:
                        continue   # behind camera

                    dist_gt  = math.sqrt(X*X + Y*Y + Z*Z)
                    s_gt_val = dist_gt / Z          # = sec(θ) from geometry

                    theta_rad = math.atan(abs(X) / Z)   # horizontal bearing

                    s_gt_list.append(s_gt_val)
                    s_pred_list.append(float(sp[pi]))
                    theta_list.append(math.degrees(theta_rad))

    return (np.array(s_gt_list),
            np.array(s_pred_list),
            np.array(theta_list))


# ── plotting ───────────────────────────────────────────────────────────────

def plot(s_gt, s_pred, theta_deg, out_path, max_s=2.2):
    # clip extreme outliers for display only
    mask = (s_gt < max_s) & (s_pred < max_s) & (s_gt >= 1.0) & (s_pred >= 0.5)
    sg   = s_gt[mask]
    sp   = s_pred[mask]
    th   = theta_deg[mask]

    fig, (ax_sc, ax_dist) = plt.subplots(
        1, 2, figsize=(8.0, 3.8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1]},
    )

    # ── (a) Scatter coloured by bearing angle ──────────────────────────────
    theta_max = np.percentile(th, 95)
    norm  = mcolors.Normalize(vmin=0, vmax=theta_max)
    cmap  = plt.cm.coolwarm

    order = np.argsort(th)
    sc = ax_sc.scatter(sg[order], sp[order],
                       c=th[order], cmap=cmap, norm=norm,
                       s=5, alpha=0.45, linewidths=0,
                       rasterized=True, zorder=3)

    lo, hi = 1.0, max_s
    ax_sc.plot([lo, hi], [lo, hi], "--", color="#333333",
               lw=1.0, alpha=0.7, zorder=4, label="$y = x$")

    ratio = np.median(sp / (sg + 1e-6))
    fit_x = np.linspace(lo, hi, 200)
    ax_sc.plot(fit_x, ratio * fit_x, "-", color="#009E73",
               lw=1.2, alpha=0.8, zorder=4,
               label=f"median $s_{{\\mathrm{{pred}}}}/s_{{\\mathrm{{gt}}}}$ = {ratio:.3f}")

    cbar = fig.colorbar(sc, ax=ax_sc, pad=0.02, fraction=0.046)
    cbar.set_label("Bearing angle θ (°)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax_sc.set_xlabel(
        r"$s_{\mathrm{gt}} = d_{\mathrm{gt}} / Z_{\mathrm{gt}}$  ($\sec\theta$)",
        fontsize=9)
    ax_sc.set_ylabel(r"$s_{\mathrm{pred}} = \exp(s_{\mathrm{raw}})$", fontsize=9)
    ax_sc.set_xlim(lo, max_s)
    ax_sc.set_ylim(lo * 0.9, max_s)
    ax_sc.set_aspect("equal", adjustable="box")
    ax_sc.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, zorder=0)
    ax_sc.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax_sc.set_title("(a) $s_\\mathrm{gt}$ vs $s_\\mathrm{pred}$", fontsize=9, pad=3)
    ax_sc.text(0.97, 0.04, f"$n = {len(sg):,}$",
               transform=ax_sc.transAxes, fontsize=7.5,
               ha="right", va="bottom",
               bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                         edgecolor="#cccccc", alpha=0.85))
    for sp_ in ("top", "right"):
        ax_sc.spines[sp_].set_visible(False)

    # ── (b) Distribution: s_gt vs s_pred ──────────────────────────────────
    bins_hist = np.linspace(1.0, max_s, 50)
    col_gt   = "#D55E00"
    col_pred = "#0072B2"

    ax_dist.hist(sg, bins=bins_hist, density=True,
                 color=col_gt,   alpha=0.55, label=r"$s_\mathrm{gt}$",
                 edgecolor="none")
    ax_dist.hist(sp, bins=bins_hist, density=True,
                 color=col_pred, alpha=0.55, label=r"$s_\mathrm{pred}$",
                 edgecolor="none")

    # KDE overlay
    from scipy.stats import gaussian_kde
    xs = np.linspace(1.0, min(max_s, 1.8), 300)
    ax_dist.plot(xs, gaussian_kde(sg)(xs),  color=col_gt,   lw=1.5)
    ax_dist.plot(xs, gaussian_kde(sp)(xs),  color=col_pred, lw=1.5)

    # vertical lines at medians — labelled via transforms so position is stable
    med_gt   = np.median(sg)
    med_pred = np.median(sp)
    ax_dist.axvline(med_gt,   color=col_gt,   lw=1.0, linestyle="--", alpha=0.8)
    ax_dist.axvline(med_pred, color=col_pred, lw=1.0, linestyle="--", alpha=0.8)
    ax_dist.text(med_gt   + 0.005, 0.72, f"med={med_gt:.3f}",
                 color=col_gt,   fontsize=7, va="top",
                 transform=ax_dist.get_xaxis_transform())
    ax_dist.text(med_pred - 0.005, 0.60, f"med={med_pred:.3f}",
                 color=col_pred, fontsize=7, va="top", ha="right",
                 transform=ax_dist.get_xaxis_transform())

    ax_dist.set_xlabel(r"$s$ value", fontsize=9)
    ax_dist.set_ylabel("Density", fontsize=9)
    ax_dist.set_xlim(1.0, 1.7)
    ax_dist.set_title(r"(b) Distribution of $s_\mathrm{gt}$ and $s_\mathrm{pred}$",
                      fontsize=9, pad=3)
    ax_dist.legend(fontsize=8, frameon=False, loc="upper right")
    ax_dist.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    for sp_ in ("top", "right"):
        ax_dist.spines[sp_].set_visible(False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", format="pdf")
    png = out_path.replace(".pdf", ".png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}  ({os.path.getsize(out_path)/1e6:.2f} MB)")
    print(f"Preview → {png}")
    print(f"\nStats (clipped to s < {max_s}):")
    print(f"  n            = {len(sg):,}  ({len(s_gt)-len(sg):,} clipped)")
    print(f"  median s_gt  = {np.median(sg):.4f}")
    print(f"  median s_pred= {np.median(sp):.4f}")
    print(f"  mae          = {np.mean(np.abs(sg - sp)):.4f}")
    print(f"  θ range      = {th.min():.1f}° – {th.max():.1f}°")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       default="runs/ogcde_lidar_p2/best.pt")
    ap.add_argument("--kitti-root", default=None)
    ap.add_argument("--split",      default="splits/distformer_val.txt")
    ap.add_argument("--out",        default="figs/sec_theta_scatter.pdf")
    ap.add_argument("--cache",      default=CACHE_PATH)
    ap.add_argument("--use-cache",  action="store_true")
    ap.add_argument("--max-s",      type=float, default=2.2,
                    help="Upper limit of both axes for plotting")
    ap.add_argument("--device",     default="cuda")
    args = ap.parse_args()

    if args.use_cache and os.path.exists(args.cache):
        z = np.load(args.cache)
        s_gt, s_pred, theta = z["s_gt"], z["s_pred"], z["theta"]
        print(f"Loaded {len(s_gt):,} pairs from cache: {args.cache}")
    else:
        if args.kitti_root is None:
            raise SystemExit("--kitti-root required unless --use-cache")
        s_gt, s_pred, theta = collect(args)
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        np.savez(args.cache, s_gt=s_gt, s_pred=s_pred, theta=theta)
        print(f"Cached {len(s_gt):,} pairs → {args.cache}")

    plot(s_gt, s_pred, theta, args.out, max_s=args.max_s)


if __name__ == "__main__":
    main()
