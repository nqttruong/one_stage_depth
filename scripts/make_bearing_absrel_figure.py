"""Fig. — AbsRel vs bearing angle: V1 (no sec-θ) vs V3 (with sec-θ).

Shows the sec(θ) correction benefit across bearing angle bins.
As objects move off-axis, direct-depth (V1, no s-head) degrades because
Z_depth < euclidean distance by factor sec(θ). V3 explicitly regresses
the sec(θ) correction via its s-head.

  V1 (--ckpt-base):  no_s_head=True  → distance estimate = exp(d_raw) ≈ Z
  V3 (--ckpt-ogcde): no_s_head=False → distance estimate = exp(d_raw + s_raw)

GT = annotation euclidean distance sqrt(X²+Y²+Z²).
θ  = arctan(|X| / Z), binned in 5° increments.

Usage:
    python scripts/make_bearing_absrel_figure.py \\
        --ckpt-base  runs/ablation/v1_baseline/best.pt \\
        --ckpt-ogcde runs/ablation/v3_sec_theta/best.pt \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt \\
        --out figs/bearing_absrel.pdf

    # re-plot from cache:
    python scripts/make_bearing_absrel_figure.py --use-cache
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

CACHE_PATH = "figs/bearing_absrel_cache.npz"
KEEP_TYPES = {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist"}

BINS       = [0, 5, 10, 15, 20, 30]
BIN_LABELS = ["0–5°", "5–10°", "10–15°", "15–20°", "20–30°"]


# ── inference helper ───────────────────────────────────────────────────────

def run_inference(args, ckpt_path):
    """Return (dist_pred, dist_gt, theta_deg) for IoU-matched pairs."""
    import torch
    from torch.utils.data import DataLoader
    from ogcde.model import OGCDENet
    from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde, parse_kitti_label
    from ogcde.utils import decode_predictions, unletterbox_boxes
    from ogcde.metrics import xywh_to_xyxy, bbox_iou_xyxy

    device = torch.device(args.device)
    ck     = torch.load(ckpt_path, map_location=device, weights_only=False)
    nc     = ck.get("args", {}).get("num_classes", 3)
    bs     = ck.get("args", {}).get("backbone_size", "n")
    no_s   = ck.get("args", {}).get("no_s_head", False)

    model = OGCDENet(nc=nc, backbone_size=bs).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    with open(args.split) as f:
        img_ids = [ln.strip() for ln in f if ln.strip()]

    label_db = {}
    for img_id in img_ids:
        p = os.path.join(args.kitti_root, "label_2", f"{img_id}.txt")
        if os.path.exists(p):
            label_db[img_id] = parse_kitti_label(p)

    ds     = KITTIOGCDEDataset(args.kitti_root, args.split,
                               img_size=640, augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False,
                        num_workers=4, collate_fn=collate_ogcde)

    dist_pred_l, dist_gt_l, theta_l = [], [], []

    with torch.no_grad():
        for imgs, targets, meta in loader:
            imgs = imgs.to(device, non_blocking=True)
            dets = decode_predictions(model(imgs), nc, obj_thr=0.25, iou_thr=0.5)

            for b, det in enumerate(dets):
                m      = meta[b]
                img_id = str(m["image_id"]).zfill(6)
                bm     = targets["batch_idx"].numpy() == b

                if not len(det["boxes"]):
                    continue

                gb  = unletterbox_boxes(
                    targets["boxes"].numpy()[bm], m["ratio"], m["pad"])
                gx  = xywh_to_xyxy(gb)
                gl  = targets["labels"].numpy()[bm]

                pb  = unletterbox_boxes(
                    det["boxes"].cpu().numpy(), m["ratio"], m["pad"])
                px  = xywh_to_xyxy(pb)

                # V1: no s-head → use depth; V3: s-head → use distance
                pred = (det["depth"] if no_s else det["distance"]).cpu().numpy()
                cls_p = det["classes"].cpu().numpy().astype(int)

                if not len(gx):
                    continue

                ious        = bbox_iou_xyxy(px, gx)
                all_objs    = label_db.get(img_id, [])
                obj_idx_arr = targets["obj_idx"].numpy()[bm]

                for pi in range(len(px)):
                    row = ious[pi].copy()
                    row[gl != cls_p[pi]] = -1
                    if row.max() < 0.5:
                        continue
                    gi = int(row.argmax())

                    oi = int(obj_idx_arr[gi]) if gi < len(obj_idx_arr) else -1
                    if oi < 0 or oi >= len(all_objs):
                        continue

                    X, Y, Z = all_objs[oi]["loc"]
                    if Z < 0.5:
                        continue

                    dist_gt  = math.sqrt(X*X + Y*Y + Z*Z)
                    theta_d  = math.degrees(math.atan(abs(X) / Z))

                    dist_pred_l.append(float(pred[pi]))
                    dist_gt_l.append(dist_gt)
                    theta_l.append(theta_d)

    return (np.array(dist_pred_l),
            np.array(dist_gt_l),
            np.array(theta_l))


# ── per-bin AbsRel ─────────────────────────────────────────────────────────

def bin_absrel(dist_gt, preds, theta_deg, bins):
    centers, means, q25s, q75s, ns = [], [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (theta_deg >= lo) & (theta_deg < hi)
        centers.append(0.5 * (lo + hi))
        n = int(m.sum())
        ns.append(n)
        if n < 10:
            means.append(np.nan); q25s.append(np.nan); q75s.append(np.nan)
        else:
            rel = np.abs(preds[m] - dist_gt[m]) / dist_gt[m]
            means.append(float(np.mean(rel) * 100))
            q25s.append(float(np.percentile(rel * 100, 25)))
            q75s.append(float(np.percentile(rel * 100, 75)))
    return (np.array(centers), np.array(means),
            np.array(q25s), np.array(q75s), ns)


# ── plot ───────────────────────────────────────────────────────────────────

def plot(base_data, ogcde_data, out_path):
    pred_b, gt_b, theta_b = base_data
    pred_o, gt_o, theta_o = ogcde_data
    bins = BINS

    xb, mb, q25b, q75b, nsb = bin_absrel(gt_b, pred_b, theta_b, bins)
    xo, mo, q25o, q75o, nso = bin_absrel(gt_o, pred_o, theta_o, bins)

    fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)

    col_base = "#D55E00"
    col_ours = "#0072B2"

    valid_b = ~np.isnan(mb)
    valid_o = ~np.isnan(mo)

    ax.fill_between(xb[valid_b], q25b[valid_b], q75b[valid_b],
                    color=col_base, alpha=0.15)
    ax.fill_between(xo[valid_o], q25o[valid_o], q75o[valid_o],
                    color=col_ours, alpha=0.15)

    ax.plot(xb[valid_b], mb[valid_b], "-o",
            color=col_base, lw=1.6, ms=5.5, zorder=4,
            label=r"V1: depth-only $\exp(d)$  [no $\sec\theta$]")
    ax.plot(xo[valid_o], mo[valid_o], "-s",
            color=col_ours, lw=1.6, ms=5.5, zorder=4,
            label=r"V3: distance $\exp(d+s)$  [$+\sec\theta$]")

    # sample-count annotations inside plot area
    for xi, ni in zip(xb[valid_b], np.array(nsb)[valid_b]):
        ax.text(xi, 0.3, f"n={ni}",
                ha="center", va="bottom", fontsize=5.5, color="#999999")

    ax.set_xticks([0.5*(lo+hi) for lo, hi in zip(bins[:-1], bins[1:])])
    ax.set_xticklabels(BIN_LABELS, fontsize=8)
    ax.set_xlabel("Bearing angle |θ|  (horizontal off-axis)", fontsize=9)
    ax.set_ylabel("AbsRel (%)", fontsize=9)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # overall AbsRel annotation
    overall_b = float(np.mean(np.abs(pred_b - gt_b) / gt_b) * 100)
    overall_o = float(np.mean(np.abs(pred_o - gt_o) / gt_o) * 100)
    ax.text(0.97, 0.95,
            f"Overall AbsRel\nV1 depth: {overall_b:.1f}%\nV3 OGCDE: {overall_o:.1f}%",
            transform=ax.transAxes, fontsize=7.5,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#cccccc", alpha=0.9))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", format="pdf")
    png = out_path.replace(".pdf", ".png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}  ({os.path.getsize(out_path)/1e6:.2f} MB)")
    print(f"Preview → {png}")

    # ── ΔAbsRel figure ──────────────────────────────────────────────────
    delta_out = out_path.replace(".pdf", "_delta.pdf")
    plot_delta(xb, mb, xo, mo, nsb, delta_out)


def plot_delta(xb, mb, xo, mo, ns, out_path):
    """Bar chart of ΔAbsRel = AbsRel(V1) − AbsRel(V3) per bin."""
    delta = mb - mo   # positive = OGCDE better

    d_max = float(np.nanmax(np.abs(delta)))
    y_top =  d_max * 1.45   # headroom above tallest positive bar
    y_bot = -d_max * 1.45

    fig, ax = plt.subplots(figsize=(4.5, 3.0), constrained_layout=True)

    colors = ["#0072B2" if d >= 0 else "#D55E00" for d in delta]
    ax.bar(xb, delta, width=3.8, color=colors, edgecolor="black", linewidth=0.4)

    # value labels: inside bar when bar is tall enough, else just outside
    for xi, d in zip(xb, delta):
        if np.isnan(d):
            continue
        bar_h = abs(d)
        inside = bar_h > d_max * 0.25   # tall enough to fit text inside
        if d >= 0:
            ypos = d - 0.02 if inside else d + 0.02
            va   = "top"    if inside else "bottom"
            col  = "white"  if inside else "black"
        else:
            ypos = d + 0.02 if inside else d - 0.02
            va   = "bottom" if inside else "top"
            col  = "white"  if inside else "black"
        ax.text(xi, ypos, f"{d:+.2f}%", ha="center", va=va,
                fontsize=8.5, fontweight="bold", color=col)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xb)
    ax.set_xticklabels(BIN_LABELS, fontsize=8)
    ax.set_xlabel("Bearing angle |θ|  (horizontal off-axis)", fontsize=9)
    ax.set_ylabel(r"$\Delta$AbsRel  [AbsRel(V1) $-$ AbsRel(V3)]  (%)", fontsize=8.5)
    ax.set_xlim(BINS[0] - 1, BINS[-1] + 1)
    ax.set_ylim(y_bot, y_top)

    # legend — lower right, away from bars
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#0072B2", edgecolor="black", linewidth=0.4,
              label="OGCDE better (+)"),
        Patch(facecolor="#D55E00", edgecolor="black", linewidth=0.4,
              label="Baseline better (−)"),
    ], fontsize=7.5, frameon=False, loc="lower right")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", format="pdf")
    png = out_path.replace(".pdf", ".png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}  ({os.path.getsize(out_path)/1e6:.2f} MB)")
    print(f"Preview → {png}")

    print("\nΔAbsRel per bin (V1 − V3, positive = OGCDE better):")
    print(f"{'Bin':12s}  {'n':>6}  {'V1 depth':>10}  {'V3 OGCDE':>10}  {'Δ':>8}")
    for i in range(len(BIN_LABELS)):
        print(f"{BIN_LABELS[i]:12s}  {ns[i]:6d}  {mb[i]:10.2f}  {mo[i]:10.2f}  "
              f"{mb[i]-mo[i]:+8.2f}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-base",  default="runs/ablation/v1_baseline/best.pt",
                    help="V1 checkpoint (no s-head, depth-only)")
    ap.add_argument("--ckpt-ogcde", default="runs/ablation/v3_sec_theta/best.pt",
                    help="V3 checkpoint (with s-head, full distance)")
    ap.add_argument("--kitti-root", default=None)
    ap.add_argument("--split",      default="splits/distformer_val.txt")
    ap.add_argument("--out",        default="figs/bearing_absrel.pdf")
    ap.add_argument("--cache",      default=CACHE_PATH)
    ap.add_argument("--use-cache",  action="store_true")
    ap.add_argument("--device",     default="cuda")
    args = ap.parse_args()

    if args.use_cache and os.path.exists(args.cache):
        z = np.load(args.cache)
        base_data  = (z["pred_b"], z["gt_b"], z["theta_b"])
        ogcde_data = (z["pred_o"], z["gt_o"], z["theta_o"])
        print(f"Loaded from cache: {args.cache}")
        print(f"  V1: {len(z['pred_b']):,} pairs | V3: {len(z['pred_o']):,} pairs")
    else:
        if args.kitti_root is None:
            raise SystemExit("--kitti-root required unless --use-cache")
        print("Running inference with V1 (baseline)...")
        base_data  = run_inference(args, args.ckpt_base)
        print(f"  → {len(base_data[0]):,} matched pairs")
        print("Running inference with V3 (sec-theta)...")
        ogcde_data = run_inference(args, args.ckpt_ogcde)
        print(f"  → {len(ogcde_data[0]):,} matched pairs")

        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        np.savez(args.cache,
                 pred_b=base_data[0], gt_b=base_data[1], theta_b=base_data[2],
                 pred_o=ogcde_data[0], gt_o=ogcde_data[1], theta_o=ogcde_data[2])
        print(f"Cached → {args.cache}")

    plot(base_data, ogcde_data, args.out)


if __name__ == "__main__":
    main()
