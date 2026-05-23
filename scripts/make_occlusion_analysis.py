"""Fig. — AbsRel by occlusion level: V1 (no CP) vs V2 (+ contact point).

Tests the hypothesis that contact point prediction helps most for occluded
objects. KITTI occlusion levels:
  0 = fully visible
  1 = partly occluded
  2 = largely occluded

V1 uses box center as the implicit spatial anchor (no contact point head).
V2 adds explicit contact point prediction, anchoring regression to the
ground-plane bottom-center — a reference invariant to top-occlusion.

Usage:
    python scripts/make_occlusion_analysis.py \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt \\
        --out figs/occlusion_analysis.pdf

    # re-plot from cache:
    python scripts/make_occlusion_analysis.py --use-cache
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

CACHE_PATH  = "figs/occlusion_cache.npz"
OCC_LABELS  = {0: "Fully visible\n(occ = 0)",
               1: "Partly occluded\n(occ = 1)",
               2: "Largely occluded\n(occ = 2)"}
KEEP_TYPES  = {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist"}


# ── parse occlusion from raw KITTI label ────────────────────────────────────

def parse_label_with_occ(path):
    """Return list of dicts with 'type', 'occ', 'loc', 'bbox', 'obj_idx'."""
    objs = []
    with open(path) as f:
        for raw_idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            objs.append({
                "type":    parts[0],
                "occ":     int(parts[2]),       # 0/1/2/3
                "loc":     [float(x) for x in parts[11:14]],
                "obj_idx": raw_idx,
            })
    return objs


# ── inference + collect pairs ───────────────────────────────────────────────

def run_inference(args, ckpt_path):
    """Return arrays: absrel, occ_level, dist_gt, dist_pred  per matched pair."""
    import torch
    from torch.utils.data import DataLoader
    from ogcde.model import OGCDENet
    from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde
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
            label_db[img_id] = parse_label_with_occ(p)

    ds     = KITTIOGCDEDataset(args.kitti_root, args.split,
                               img_size=640, augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False,
                        num_workers=4, collate_fn=collate_ogcde)

    absrel_l, occ_l, dist_gt_l, dist_pred_l = [], [], [], []

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
                    gi = int(row.argmax())

                    oi = int(obj_idx_arr[gi]) if gi < len(obj_idx_arr) else -1
                    if oi < 0 or oi >= len(all_objs):
                        continue

                    obj = all_objs[oi]
                    if obj["type"] not in KEEP_TYPES:
                        continue

                    occ = obj["occ"]
                    if occ == 3:          # unknown — skip
                        continue

                    X, Y, Z = obj["loc"]
                    if Z < 0.5:
                        continue

                    dist_gt  = math.sqrt(X*X + Y*Y + Z*Z)
                    ar       = abs(float(pred[pi]) - dist_gt) / dist_gt

                    absrel_l.append(ar)
                    occ_l.append(occ)
                    dist_gt_l.append(dist_gt)
                    dist_pred_l.append(float(pred[pi]))

    return (np.array(absrel_l),
            np.array(occ_l, dtype=np.int32),
            np.array(dist_gt_l),
            np.array(dist_pred_l))


# ── plot ────────────────────────────────────────────────────────────────────

def plot(data_v1, data_v2, out_path):
    ar1, occ1 = data_v1[:2]
    ar2, occ2 = data_v2[:2]

    occ_levels = [0, 1, 2]
    col_v1 = "#D55E00"
    col_v2 = "#0072B2"
    width  = 0.32
    x      = np.arange(len(occ_levels))

    fig, (ax_bar, ax_delta) = plt.subplots(
        1, 2, figsize=(8.0, 3.8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.4, 1]},
    )

    # ── (a) grouped bar chart ────────────────────────────────────────────
    means_v1, means_v2, ns_v1, ns_v2 = [], [], [], []
    for occ in occ_levels:
        m1 = ar1[occ1 == occ]
        m2 = ar2[occ2 == occ]
        means_v1.append(float(np.mean(m1) * 100) if len(m1) >= 5 else np.nan)
        means_v2.append(float(np.mean(m2) * 100) if len(m2) >= 5 else np.nan)
        ns_v1.append(len(m1))
        ns_v2.append(len(m2))

    bars1 = ax_bar.bar(x - width/2, means_v1, width,
                       color=col_v1, edgecolor="black", linewidth=0.4,
                       label="V1: no contact point")
    bars2 = ax_bar.bar(x + width/2, means_v2, width,
                       color=col_v2, edgecolor="black", linewidth=0.4,
                       label="V2: + contact point")

    for bar, val in zip(list(bars1) + list(bars2), means_v1 + means_v2):
        if np.isnan(val):
            continue
        ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([OCC_LABELS[o] for o in occ_levels], fontsize=8)
    ax_bar.set_ylabel("AbsRel (%)", fontsize=9)
    ax_bar.set_title("(a) AbsRel by occlusion level", fontsize=9, pad=3)
    ax_bar.set_ylim(0, max([v for v in means_v1 + means_v2 if not np.isnan(v)]) * 1.3)
    ax_bar.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax_bar.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax_bar.spines[sp].set_visible(False)

    # ── (b) ΔAbsRel = V1 - V2 ────────────────────────────────────────────
    delta = [v1 - v2 for v1, v2 in zip(means_v1, means_v2)]
    colors_d = ["#0072B2" if d >= 0 else "#D55E00" for d in delta]

    bars_d = ax_delta.bar(x, delta, width=0.5, color=colors_d,
                          edgecolor="black", linewidth=0.4)

    for bar, d in zip(bars_d, delta):
        if np.isnan(d):
            continue
        # always label outside the bar to avoid clipping
        if d >= 0:
            ypos, va = d + 0.02, "bottom"
        else:
            ypos, va = d - 0.02, "top"
        ax_delta.text(bar.get_x() + bar.get_width()/2, ypos,
                      f"{d:+.2f}pp", ha="center", va=va,
                      fontsize=9, fontweight="bold", color="black")

    ax_delta.axhline(0, color="black", lw=0.8)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels([OCC_LABELS[o] for o in occ_levels], fontsize=8)
    ax_delta.set_ylabel(r"$\Delta$AbsRel  [V1 $-$ V2]  (pp)", fontsize=8.5)
    ax_delta.set_title("(b) Improvement from contact point", fontsize=9, pad=3)
    d_max = max(abs(v) for v in delta if not np.isnan(v))
    ax_delta.set_ylim(-d_max * 1.6, d_max * 1.6)
    ax_delta.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax_delta.spines[sp].set_visible(False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", format="pdf")
    png = out_path.replace(".pdf", ".png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}  ({os.path.getsize(out_path)/1e6:.2f} MB)")
    print(f"Preview → {png}")

    print("\nPer-occlusion summary:")
    print(f"{'Level':25s}  {'n(V1)':>6}  {'V1':>8}  {'n(V2)':>6}  {'V2':>8}  {'Δ(pp)':>8}")
    for i, occ in enumerate(occ_levels):
        print(f"{OCC_LABELS[occ].replace(chr(10),' '):25s}  "
              f"{ns_v1[i]:6d}  {means_v1[i]:8.2f}  "
              f"{ns_v2[i]:6d}  {means_v2[i]:8.2f}  {delta[i]:+8.2f}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-v1",    default="runs/ablation/v1_baseline/best.pt")
    ap.add_argument("--ckpt-v2",    default="runs/ablation/v2_cp/best.pt")
    ap.add_argument("--kitti-root", default=None)
    ap.add_argument("--split",      default="splits/distformer_val.txt")
    ap.add_argument("--out",        default="figs/occlusion_analysis.pdf")
    ap.add_argument("--cache",      default=CACHE_PATH)
    ap.add_argument("--use-cache",  action="store_true")
    ap.add_argument("--device",     default="cuda")
    args = ap.parse_args()

    if args.use_cache and os.path.exists(args.cache):
        z = np.load(args.cache)
        data_v1 = (z["ar1"], z["occ1"], z["gt1"], z["pred1"])
        data_v2 = (z["ar2"], z["occ2"], z["gt2"], z["pred2"])
        print(f"Loaded from cache: V1 {len(z['ar1']):,} | V2 {len(z['ar2']):,} pairs")
    else:
        if args.kitti_root is None:
            raise SystemExit("--kitti-root required unless --use-cache")
        print("Running inference V1 (no CP)...")
        data_v1 = run_inference(args, args.ckpt_v1)
        print(f"  → {len(data_v1[0]):,} matched pairs")
        print("Running inference V2 (+ CP)...")
        data_v2 = run_inference(args, args.ckpt_v2)
        print(f"  → {len(data_v2[0]):,} matched pairs")
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        np.savez(args.cache,
                 ar1=data_v1[0], occ1=data_v1[1], gt1=data_v1[2], pred1=data_v1[3],
                 ar2=data_v2[0], occ2=data_v2[1], gt2=data_v2[2], pred2=data_v2[3])
        print(f"Cached → {args.cache}")

    plot(data_v1, data_v2, args.out)


if __name__ == "__main__":
    main()
