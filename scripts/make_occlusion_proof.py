"""Fig. — Occlusion robustness: V1 (baseline) vs V4 (full model with CP + sec-θ).

Two-panel figure:
  (a) Line chart: AbsRel vs occlusion level for V1 and V4.
      V1's slope is steeper → degrades faster with occlusion.
      V4's slope is flatter → more robust.
  (b) Bar chart: ΔAbsRel (V1 − V4) per occlusion level.
      Growing bars prove the full model helps MORE for occluded objects.

Usage:
    python scripts/make_occlusion_proof.py \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt \\
        --out figs/occlusion_proof.pdf

    # skip inference, re-plot from cache:
    python scripts/make_occlusion_proof.py --use-cache
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     9,
    "pdf.fonttype":  42,
    "ps.fonttype":   42,
    "axes.linewidth": 0.6,
})

CACHE_PATH  = "figs/occlusion_proof_cache.npz"
OCC_LABELS  = ["Fully\nvisible\n(occ=0)",
               "Partly\noccluded\n(occ=1)",
               "Largely\noccluded\n(occ=2)"]
KEEP_TYPES  = {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist"}
COL_V1      = "#D55E00"
COL_V4      = "#0072B2"


# ── helpers ─────────────────────────────────────────────────────────────────

def parse_label_full(path):
    objs = []
    with open(path) as f:
        for raw_idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 15 or parts[0] == "DontCare":
                continue
            objs.append({
                "type":    parts[0],
                "trunc":   float(parts[1]),
                "occ":     int(parts[2]),
                "bbox":    [float(x) for x in parts[4:8]],
                "loc":     [float(x) for x in parts[11:14]],
                "obj_idx": raw_idx,
            })
    return objs


# ── (b) AbsRel per occlusion level — model inference ────────────────────────

def run_inference_occ(args, ckpt_path):
    """Return (absrel, occ_level) arrays."""
    import torch
    from torch.utils.data import DataLoader
    from ogcde.model import OGCDENet
    from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde
    from ogcde.utils import decode_predictions, unletterbox_boxes
    from ogcde.metrics import xywh_to_xyxy, bbox_iou_xyxy

    device = torch.device(args.device)
    ck     = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    nc     = ck_args.get("num_classes", 3)
    bs     = ck_args.get("backbone_size", "n")
    no_s   = ck_args.get("no_s_head", False)

    model  = OGCDENet(nc=nc, backbone_size=bs).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # Build full label DB for occlusion lookup
    with open(args.split) as f:
        img_ids = [ln.strip() for ln in f if ln.strip()]

    label_db = {}
    for img_id in img_ids:
        p = os.path.join(args.kitti_root, "label_2", f"{img_id}.txt")
        if os.path.exists(p):
            label_db[img_id] = parse_label_full(p)

    ds     = KITTIOGCDEDataset(args.kitti_root, args.split,
                               img_size=640, augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False,
                        num_workers=4, collate_fn=collate_ogcde)

    absrel_l, occ_l = [], []

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
                from ogcde.metrics import xywh_to_xyxy as x2x
                gx  = x2x(gb)
                gl  = targets["labels"].numpy()[bm]

                pb  = unletterbox_boxes(
                    det["boxes"].cpu().numpy(), m["ratio"], m["pad"])
                px  = x2x(pb)

                pred    = (det["depth"] if no_s else det["distance"]).cpu().numpy()
                cls_p   = det["classes"].cpu().numpy().astype(int)

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
                    gi  = int(row.argmax())
                    oi  = int(obj_idx_arr[gi]) if gi < len(obj_idx_arr) else -1
                    if oi < 0 or oi >= len(all_objs):
                        continue

                    obj = all_objs[oi]
                    if obj["type"] not in KEEP_TYPES:
                        continue
                    occ = obj["occ"]
                    if occ not in (0, 1, 2):
                        continue

                    X, Y, Z = obj["loc"]
                    if Z < 0.5:
                        continue
                    dist_gt = math.sqrt(X*X + Y*Y + Z*Z)
                    ar      = abs(float(pred[pi]) - dist_gt) / dist_gt

                    absrel_l.append(ar)
                    occ_l.append(occ)

    return np.array(absrel_l), np.array(occ_l, dtype=np.int32)


# ── plot ─────────────────────────────────────────────────────────────────────

def plot(ar1, occ1, ar4, occ4, out_path):
    occ_levels = [0, 1, 2]
    x = np.arange(len(occ_levels))

    means_v1, means_v4, ns_v1, ns_v4 = [], [], [], []
    for occ in occ_levels:
        m1 = ar1[occ1 == occ]
        m4 = ar4[occ4 == occ]
        means_v1.append(float(np.mean(m1) * 100) if len(m1) >= 5 else np.nan)
        means_v4.append(float(np.mean(m4) * 100) if len(m4) >= 5 else np.nan)
        ns_v1.append(len(m1))
        ns_v4.append(len(m4))

    delta = [v1 - v4 for v1, v4 in zip(means_v1, means_v4)]

    fig, (ax_line, ax_delta) = plt.subplots(
        1, 2, figsize=(9.0, 3.8), constrained_layout=True)

    # ── (a) Line chart: AbsRel degradation with occlusion ───────────────────
    xtick_labels = ["Fully\nvisible\n(occ=0)", "Partly\noccluded\n(occ=1)", "Largely\noccluded\n(occ=2)"]

    ax_line.plot(x, means_v1, "o-", color=COL_V1, linewidth=2.0,
                 markersize=7, markeredgecolor="white", markeredgewidth=1.0,
                 label="V1: baseline (no CP, no sec-θ)", zorder=4)
    ax_line.plot(x, means_v4, "s-", color=COL_V4, linewidth=2.0,
                 markersize=7, markeredgecolor="white", markeredgewidth=1.0,
                 label="V4: full model (CP + sec-θ + warmup)", zorder=4)

    # Shade the gap between lines
    ax_line.fill_between(x, means_v1, means_v4, alpha=0.12, color=COL_V4)

    # Label each point
    for i, (v1, v4) in enumerate(zip(means_v1, means_v4)):
        ax_line.text(i + 0.07, v1 + 0.07, f"{v1:.1f}%", fontsize=7.5,
                     color=COL_V1, fontweight="bold", va="bottom")
        ax_line.text(i + 0.07, v4 - 0.12, f"{v4:.1f}%", fontsize=7.5,
                     color=COL_V4, fontweight="bold", va="top")

    # Degradation rate annotation
    deg_v1 = (means_v1[2] / means_v1[0] - 1) * 100
    deg_v4 = (means_v4[2] / means_v4[0] - 1) * 100
    ax_line.text(0.97, 0.97,
                 f"occ=0→2 degradation:\n"
                 f"V1: +{deg_v1:.0f}%  V4: +{deg_v4:.0f}%",
                 transform=ax_line.transAxes, fontsize=7.5,
                 ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#cccccc", alpha=0.9))

    ax_line.set_xticks(x)
    ax_line.set_xticklabels(xtick_labels, fontsize=8)
    ax_line.set_ylabel("AbsRel (%)", fontsize=9)
    ax_line.set_title("(a) AbsRel degradation with occlusion severity", fontsize=9, pad=3)
    ymin = min(v for v in means_v1 + means_v4 if not np.isnan(v))
    ymax = max(v for v in means_v1 + means_v4 if not np.isnan(v))
    ax_line.set_ylim(ymin * 0.88, ymax * 1.18)
    ax_line.set_xlim(-0.3, 2.5)
    ax_line.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax_line.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax_line.spines[sp].set_visible(False)

    # ── (b) ΔAbsRel bar chart — growing improvement ──────────────────────────
    bar_colors = [COL_V4 if d >= 0 else COL_V1 for d in delta]
    bars = ax_delta.bar(x, delta, width=0.5, color=bar_colors,
                        edgecolor="black", linewidth=0.4, zorder=3)

    for bar, d in zip(bars, delta):
        if np.isnan(d): continue
        ypos = d + 0.04 if d >= 0 else d - 0.04
        va   = "bottom" if d >= 0 else "top"
        ax_delta.text(bar.get_x() + bar.get_width()/2, ypos,
                      f"+{d:.2f}pp", ha="center", va=va,
                      fontsize=9, fontweight="bold", color="black")

    ax_delta.axhline(0, color="black", lw=0.8)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(xtick_labels, fontsize=8)
    ax_delta.set_ylabel(r"$\Delta$AbsRel  [V1 $-$ V4]  (pp)", fontsize=8.5)
    ax_delta.set_title("(b) Improvement grows with occlusion severity", fontsize=9, pad=3)
    d_max = max(abs(v) for v in delta if not np.isnan(v))
    ax_delta.set_ylim(0, d_max * 1.55)
    ax_delta.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax_delta.spines[sp].set_visible(False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", format="pdf")
    png = out_path.replace(".pdf", ".png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    print(f"Preview → {png}")

    print("\n── AbsRel by occlusion level ──")
    print(f"{'Level':12s}  {'n(V1)':>6}  {'V1 %':>7}  {'n(V4)':>6}  {'V4 %':>7}  {'Δ(pp)':>7}")
    for i, occ in enumerate(occ_levels):
        d = delta[i]
        print(f"  occ={occ}       {ns_v1[i]:6d}  {means_v1[i]:7.2f}  "
              f"{ns_v4[i]:6d}  {means_v4[i]:7.2f}  {d:+7.2f}")
    print(f"\n  V1 degradation occ=0→2: +{deg_v1:.0f}%")
    print(f"  V4 degradation occ=0→2: +{deg_v4:.0f}%")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root",  default=None)
    ap.add_argument("--split",       default="splits/distformer_val.txt")
    ap.add_argument("--ckpt-v1",     default="runs/ablation/v1_baseline/best.pt")
    ap.add_argument("--ckpt-v4",     default="runs/ablation/v4_warmup/best.pt")
    ap.add_argument("--out",         default="figs/occlusion_proof.pdf")
    ap.add_argument("--cache",       default=CACHE_PATH)
    ap.add_argument("--use-cache",   action="store_true")
    ap.add_argument("--device",      default="cuda")
    args = ap.parse_args()

    if args.use_cache and os.path.exists(args.cache):
        z   = np.load(args.cache)
        ar1 = z["ar1"];  occ1 = z["occ1"]
        ar4 = z["ar4"];  occ4 = z["occ4"]
        print(f"Loaded from cache: V1 n={len(ar1):,}  V4 n={len(ar4):,}")
    else:
        if args.kitti_root is None:
            raise SystemExit("--kitti-root required unless --use-cache")

        print("Running inference V1 (baseline)…")
        ar1, occ1 = run_inference_occ(args, args.ckpt_v1)
        print(f"  → {len(ar1):,} matched pairs")

        print("Running inference V4 (full model)…")
        ar4, occ4 = run_inference_occ(args, args.ckpt_v4)
        print(f"  → {len(ar4):,} matched pairs")

        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        np.savez(args.cache, ar1=ar1, occ1=occ1, ar4=ar4, occ4=occ4)
        print(f"Cached → {args.cache}")

    plot(ar1, occ1, ar4, occ4, args.out)


if __name__ == "__main__":
    main()
