"""Evaluate all ablation variants + full model on Chen val split.

Outputs:
  results/ablation.csv        — variant, abs_rel, delta_1_25, n_matched
  results/ablation_table.tex  — booktabs LaTeX table

Usage:
    python scripts/eval_ablation.py \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt \\
        --lidar-gt cache/lidar_gt_val.json
"""

import argparse
import json
import math
import os
import textwrap

import numpy as np
import torch
from torch.utils.data import DataLoader

from ogcde.model import OGCDENet
from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde
from ogcde.utils import decode_predictions, unletterbox_boxes
from ogcde.metrics import OGCDEEvaluator

# (variant_id, display_name, checkpoint_path)
VARIANTS = [
    ("v1_baseline",  r"V1: det + direct depth",             "runs/ablation/v1_baseline/best.pt"),
    ("v2_cp",        r"V2: + contact point",                "runs/ablation/v2_cp/best.pt"),
    ("v3_sec_theta", r"V3: + $\sec\theta$ head",            "runs/ablation/v3_sec_theta/best.pt"),
    # V4: use last.pt — best.pt is epoch=0 (warmup makes early val_loss artificially low)
    ("v4_warmup",    r"V4: + $\lambda_\mathrm{geo}$ warmup","runs/ablation/v4_warmup/last.pt"),
    # V5 (+ per-class weights = full OGCDE) reported in Table I — see RESULTS.md
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root",  required=True)
    ap.add_argument("--split",       default="splits/distformer_val.txt")
    ap.add_argument("--lidar-gt",    default="cache/lidar_gt_val.json")
    ap.add_argument("--img-size",    type=int,   default=640)
    ap.add_argument("--batch",       type=int,   default=16)
    ap.add_argument("--workers",     type=int,   default=4)
    ap.add_argument("--conf",        type=float, default=0.25)
    ap.add_argument("--iou-thr",     type=float, default=0.5)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir",     default="results")
    return ap.parse_args()


def evaluate_checkpoint(ckpt_path, args, lidar_db):
    device = torch.device(args.device)

    ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_args = ckpt.get("args", {})
    bs      = ck_args.get("backbone_size", "n")
    nc      = ck_args.get("num_classes", 3)
    epoch   = ckpt.get("epoch", -1)
    # For variants without s_head, distance = depth (s_raw untrained → exp(d+s) garbage)
    use_depth_as_dist = ck_args.get("no_s_head", False)

    model = OGCDENet(nc=nc, backbone_size=bs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = KITTIOGCDEDataset(
        args.kitti_root, args.split,
        img_size=args.img_size, augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_ogcde)

    # collect matched pairs
    dist_pred_all, dist_gt_all = [], []

    with torch.no_grad():
        for imgs, targets, meta in loader:
            imgs  = imgs.to(device, non_blocking=True)
            preds = model(imgs)
            dets  = decode_predictions(preds, nc,
                                       obj_thr=args.conf, iou_thr=args.iou_thr)

            for b, det in enumerate(dets):
                m      = meta[b]
                img_id = m["image_id"]
                bmask  = targets["batch_idx"].numpy() == b

                gt_boxes  = targets["boxes"].numpy()[bmask]
                gt_labels = targets["labels"].numpy()[bmask]
                gt_obj    = targets["obj_idx"].numpy()[bmask]

                from ogcde.metrics import xywh_to_xyxy, bbox_iou_xyxy
                pred_boxes = unletterbox_boxes(det["boxes"].cpu().numpy(), m["ratio"], m["pad"])
                pred_xyxy  = xywh_to_xyxy(pred_boxes)
                gt_xyxy    = xywh_to_xyxy(
                    unletterbox_boxes(gt_boxes, m["ratio"], m["pad"]))
                # V1/V2: s_raw untrained → use depth (exp d_raw) as distance proxy
                pred_dist = (det["depth"] if use_depth_as_dist
                             else det["distance"]).cpu().numpy()
                pred_cls   = det["classes"].cpu().numpy()
                pred_score = det["scores"].cpu().numpy()
                gt_labels_arr = gt_labels

                if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
                    continue

                ious = bbox_iou_xyxy(pred_xyxy, gt_xyxy)
                lidar_recs = {r["obj_idx"]: r for r in lidar_db.get(str(img_id), [])}

                for pi in range(len(pred_xyxy)):
                    c = int(pred_cls[pi])
                    row = ious[pi].copy()
                    row[gt_labels_arr != c] = -1
                    if row.max() < args.iou_thr:
                        continue
                    gi = int(row.argmax())
                    raw_idx = int(gt_obj[gi])
                    lr = lidar_recs.get(raw_idx)
                    if lr is None or lr.get("n_points", 0) == 0:
                        continue
                    dist_pred_all.append(float(pred_dist[pi]))
                    dist_gt_all.append(float(lr["dist_lidar"]))

    dist_pred = np.array(dist_pred_all)
    dist_gt   = np.array(dist_gt_all)
    n = len(dist_pred)
    if n == 0:
        return {"abs_rel": float("nan"), "delta_1_25": float("nan"),
                "n_matched": 0, "epoch": epoch}

    abs_rel    = float(np.mean(np.abs(dist_pred - dist_gt) / dist_gt))
    ratio      = np.maximum(dist_pred / dist_gt, dist_gt / dist_pred)
    delta_1_25 = float(np.mean(ratio < 1.25))
    return {"abs_rel": abs_rel, "delta_1_25": delta_1_25,
            "n_matched": n, "epoch": epoch}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.lidar_gt) as f:
        lidar_db = json.load(f)

    rows = []
    for vid, name, ckpt_path in VARIANTS:
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {vid}: {ckpt_path} not found")
            rows.append({"variant": vid, "name": name,
                         "abs_rel": float("nan"), "delta_1_25": float("nan"),
                         "n_matched": 0, "epoch": -1})
            continue

        print(f"Evaluating {vid} ({ckpt_path}) ...", flush=True)
        res = evaluate_checkpoint(ckpt_path, args, lidar_db)
        print(f"  epoch={res['epoch']}  AbsRel={res['abs_rel']*100:.2f}%  "
              f"δ<1.25={res['delta_1_25']*100:.2f}%  n={res['n_matched']}")
        rows.append({"variant": vid, "name": name, **res})

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "ablation.csv")
    with open(csv_path, "w") as f:
        f.write("variant,name,abs_rel,delta_1_25,n_matched,epoch\n")
        for r in rows:
            f.write(f"{r['variant']},{r['name']},"
                    f"{r['abs_rel']:.6f},{r['delta_1_25']:.6f},"
                    f"{r['n_matched']},{r['epoch']}\n")
    print(f"\nSaved → {csv_path}")

    # ── monotonicity check ────────────────────────────────────────────────
    valid = [r for r in rows if not math.isnan(r["abs_rel"])]
    abs_rels = [r["abs_rel"] for r in valid]
    if abs_rels != sorted(abs_rels):
        print("\n⚠️  WARNING: AbsRel is NOT monotonically decreasing V1→V5!")
        print("   Values:", [f"{x*100:.2f}%" for x in abs_rels])
        print("   Stop and diagnose before reporting in paper.")
    else:
        print("\n✅ AbsRel monotonically decreases V1→V5 — story holds.")

    # ── LaTeX table ──────────────────────────────────────────────────────
    tex_path = os.path.join(args.out_dir, "ablation_table.tex")
    def fmt(v, pct=True):
        if math.isnan(v): return r"\textemdash"
        return f"{v*100:.2f}" if pct else f"{v:.4f}"

    tex = textwrap.dedent(r"""
        \begin{table}[t]
        \centering
        \caption{Ablation study (100-epoch protocol, Chen val split, LiDAR 10th-pct GT).}
        \label{tab:ablation}
        \begin{tabular}{lcc}
        \toprule
        Variant & AbsRel $\downarrow$ & $\delta{<}1.25$ $\uparrow$ \\
        \midrule
    """).strip() + "\n"

    for r in rows:
        name    = r["name"].replace("_", r"\_")
        absrel  = fmt(r["abs_rel"])
        delta   = fmt(r["delta_1_25"])
        marker  = r" \quad\textbf{(full)}" if r["variant"] == "v5_full" else ""
        tex += f"        {name} & {absrel}\\% & {delta}\\% \\\\\n"

    # note about V5
    tex += "        \\midrule\n"
    tex += "        V5: + per-class weights (full OGCDE) & \\multicolumn{2}{c}{see Table~\\ref{tab:main}} \\\\\n"

    tex += textwrap.dedent(r"""
        \bottomrule
        \end{tabular}
        \end{table}
    """).strip() + "\n"

    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Saved → {tex_path}")


if __name__ == "__main__":
    main()
