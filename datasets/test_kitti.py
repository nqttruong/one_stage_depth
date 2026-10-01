"""Full test evaluation of OGCDE on KITTI test split.

Outputs in --out-dir (default runs/test_v7/):
  vis/           ← annotated images (pred boxes + GT boxes)
  plots/         ← scatter, histogram, PR curves, class stats
  metrics.json   ← summary metrics

Usage:
    python test_kitti.py \
        --ckpt runs/ogcde_v7/best.pt \
        --kitti-root /data/kitti/training \
        --split splits/kitti_test.txt \
        --out-dir runs/test_v7
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from ogcde.model import OGCDENet
from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde
from ogcde.utils import decode_predictions, letterbox, unletterbox_boxes, unletterbox_points
from ogcde.metrics import (
    OGCDEEvaluator, MAPEvaluator,
    xywh_to_xyxy, greedy_match, bbox_iou_xyxy,
)

CLASS_NAMES  = ["Vehicle", "Pedestrian", "Cyclist"]
CLASS_COLORS = [(0, 200, 0), (200, 80, 0), (0, 140, 255)]   # BGR
GT_COLOR     = (0, 220, 220)


# ──────────────────────────── visualisation ───────────────────────────────────

def draw_boxes(img, boxes_xywh, labels=None, distances=None,
               color=(0, 255, 0), gt=False, scores=None):
    """Draw bounding boxes on img (in-place). boxes_xywh: (N,4) original coords."""
    h, w = img.shape[:2]
    for i, box in enumerate(boxes_xywh):
        cx, cy, bw, bh = box
        x1 = int(cx - bw / 2); y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2); y2 = int(cy + bh / 2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        c = GT_COLOR if gt else (CLASS_COLORS[int(labels[i])] if labels is not None else color)
        thick = 1 if gt else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thick)
        if not gt and labels is not None:
            parts = [CLASS_NAMES[int(labels[i])]]
            if scores  is not None: parts.append(f"{scores[i]:.2f}")
            if distances is not None: parts.append(f"{distances[i]:.1f}m")
            txt = " ".join(parts)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(y1 - 4, th + 2)
            cv2.rectangle(img, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), c, -1)
            cv2.putText(img, txt, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return img


def make_vis(bgr_img, pred_boxes, pred_labels, pred_scores, pred_dist,
             gt_boxes, gt_labels):
    vis = bgr_img.copy()
    # GT boxes (thin yellow)
    if len(gt_boxes):
        draw_boxes(vis, gt_boxes, gt_labels, gt=True)
    # Pred boxes
    if len(pred_boxes):
        draw_boxes(vis, pred_boxes, pred_labels, pred_dist, scores=pred_scores)
    # legend
    cv2.putText(vis, "GT", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GT_COLOR, 1)
    cv2.putText(vis, "Pred", (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return vis


# ──────────────────────────── inference ───────────────────────────────────────

def infer(model, bgr_img, img_size, obj_thr, iou_thr, device, nc):
    img, ratio, pad = letterbox(bgr_img, img_size)
    inp = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model(inp)
    dets = decode_predictions(raw, nc, obj_thr=obj_thr, iou_thr=iou_thr)[0]
    boxes   = unletterbox_boxes(dets["boxes"].cpu().numpy(), ratio, pad)
    contact = unletterbox_points(dets["contact"].cpu().numpy(), ratio, pad)
    return {
        "boxes":    boxes,
        "scores":   dets["scores"].cpu().numpy(),
        "classes":  dets["classes"].cpu().numpy(),
        "distance": dets["distance"].cpu().numpy(),
        "depth":    dets["depth"].cpu().numpy(),
        "contact":  contact,
    }


# ──────────────────────────── plots ───────────────────────────────────────────

def plot_scatter(records, out_path):
    """Scatter: dist_pred vs dist_gt, colour by class."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    for ci, (cname, col) in enumerate(zip(CLASS_NAMES, colors)):
        sub = [r for r in records if r["cls"] == ci]
        if not sub: continue
        dp = [r["dist_pred"] for r in sub]
        dg = [r["dist_gt"]   for r in sub]
        ax.scatter(dg, dp, c=col, s=8, alpha=0.5, label=cname)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    ax.set_xlabel("GT distance (m)"); ax.set_ylabel("Pred distance (m)")
    ax.set_title("Distance prediction vs Ground-truth")
    ax.legend(markerscale=2); ax.set_aspect("equal", "box")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_error_hist(records, out_path):
    """Histogram of per-sample absolute relative error."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]

    # AbsRel histogram
    all_rel = [abs(r["dist_pred"] - r["dist_gt"]) / r["dist_gt"] for r in records if r["dist_gt"] > 0]
    axes[0].hist(all_rel, bins=40, color="steelblue", edgecolor="white", linewidth=0.4)
    axes[0].axvline(np.mean(all_rel), color="red", ls="--", lw=1.5, label=f"mean={np.mean(all_rel):.4f}")
    axes[0].set_xlabel("AbsRel"); axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of per-sample AbsRel"); axes[0].legend()

    # Error vs GT distance
    dg = [r["dist_gt"] for r in records if r["dist_gt"] > 0]
    ae = [abs(r["dist_pred"] - r["dist_gt"]) for r in records if r["dist_gt"] > 0]
    axes[1].scatter(dg, ae, s=5, alpha=0.3, c="steelblue")
    axes[1].set_xlabel("GT distance (m)"); axes[1].set_ylabel("|pred - GT| (m)")
    axes[1].set_title("Absolute error vs distance")

    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_pr_curves(preds_all, gts_all, n_gt_per_class, iou_thr, out_path):
    """PR curves per class (from raw per-image pred/GT collected during inference)."""
    from ogcde.metrics import compute_ap
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    for ci, (cname, col, ax) in enumerate(zip(CLASS_NAMES, colors, axes)):
        preds_c = [(s, iid, pb) for s, iid, pb, pc in preds_all if pc == ci]
        n_gt = n_gt_per_class[ci]
        if n_gt == 0 or len(preds_c) == 0:
            ax.set_title(f"{cname}\n(no data)"); continue
        preds_c.sort(key=lambda x: -x[0])
        matched = {}
        tp = np.zeros(len(preds_c)); fp = np.zeros(len(preds_c))
        for i, (score, img_id, pbox) in enumerate(preds_c):
            key = (img_id, ci)
            gt_boxes = gts_all.get(key, np.zeros((0, 4)))
            if len(gt_boxes) == 0: fp[i] = 1; continue
            if key not in matched:
                matched[key] = np.zeros(len(gt_boxes), dtype=bool)
            ious = bbox_iou_xyxy(pbox[None], gt_boxes)[0]
            ious[matched[key]] = -1
            best = int(np.argmax(ious))
            if ious[best] >= iou_thr:
                tp[i] = 1; matched[key][best] = True
            else:
                fp[i] = 1
        cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
        rec  = cum_tp / (n_gt + 1e-9)
        prec = cum_tp / (cum_tp + cum_fp + 1e-9)
        ap = compute_ap(rec, prec)
        ax.plot(rec, prec, color=col, lw=1.5)
        ax.fill_between(rec, prec, alpha=0.15, color=col)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"{cname}  AP={ap:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Precision-Recall curves (IoU≥{iou_thr})", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_class_stats(records, tp_counts, fp_counts, fn_counts, ap_list, ap3d_list, out_path):
    """Bar charts: TP/FP/FN counts + AP vs AP3D per class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(CLASS_NAMES)); w = 0.25
    axes[0].bar(x - w, tp_counts, w, label="TP", color="#2ca02c")
    axes[0].bar(x,     fp_counts, w, label="FP", color="#d62728")
    axes[0].bar(x + w, fn_counts, w, label="FN (missed)", color="#aec7e8")
    axes[0].set_xticks(x); axes[0].set_xticklabels(CLASS_NAMES)
    axes[0].set_ylabel("Count"); axes[0].set_title("Detection breakdown per class")
    axes[0].legend()

    w2 = 0.35
    valid_ap    = [a if not np.isnan(a) else 0 for a in ap_list]
    valid_ap3d  = [a if not np.isnan(a) else 0 for a in ap3d_list]
    axes[1].bar(x - w2/2, valid_ap,   w2, label="AP@0.5",        color="#1f77b4")
    axes[1].bar(x + w2/2, valid_ap3d, w2, label="AP3D@0.5+δ1",   color="#ff7f0e")
    for xi, (a, b) in enumerate(zip(valid_ap, valid_ap3d)):
        axes[1].text(xi - w2/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        axes[1].text(xi + w2/2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_ylim(0, 1.1); axes[1].set_ylabel("AP")
    axes[1].set_title("AP vs AP3D per class"); axes[1].legend()

    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_dist_distribution(records, out_path):
    """Histogram of GT distance distribution per class + overall pred vs GT."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]

    # Per-class GT distance histograms
    bins = np.linspace(0, 70, 36)
    for ci, (cname, col) in enumerate(zip(CLASS_NAMES, colors)):
        dg = [r["dist_gt"] for r in records if r["cls"] == ci and r["dist_gt"] > 0]
        if dg:
            axes[0].hist(dg, bins=bins, alpha=0.6, color=col, label=f"{cname} (n={len(dg)})")
    axes[0].set_xlabel("Distance (m)"); axes[0].set_ylabel("Count")
    axes[0].set_title("GT distance distribution per class"); axes[0].legend()

    # Overall pred vs GT distance overlay
    dg_all = [r["dist_gt"]   for r in records if r["dist_gt"] > 0]
    dp_all = [r["dist_pred"] for r in records if r["dist_gt"] > 0]
    axes[1].hist(dg_all, bins=bins, alpha=0.5, color="steelblue", label="GT")
    axes[1].hist(dp_all, bins=bins, alpha=0.5, color="orange",    label="Pred")
    axes[1].set_xlabel("Distance (m)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Pred vs GT distance distribution (all matched)")
    axes[1].legend()

    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_ale_vs_dist(records, out_path):
    """ALE vs GT distance scatter."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    for ci, (cname, col) in enumerate(zip(CLASS_NAMES, colors)):
        sub = [r for r in records if r["cls"] == ci and "ale" in r]
        if not sub: continue
        dg  = [r["dist_gt"] for r in sub]
        ale = [r["ale"]     for r in sub]
        ax.scatter(dg, ale, c=col, s=8, alpha=0.5, label=cname)
    ax.set_xlabel("GT distance (m)"); ax.set_ylabel("ALE (m)")
    ax.set_title("3D Localization Error vs distance"); ax.legend(markerscale=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ──────────────────────────── main ────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--split",      required=True)
    ap.add_argument("--out-dir",    default="runs/test_v7")
    ap.add_argument("--img-size",   type=int,   default=640)
    ap.add_argument("--batch",      type=int,   default=1)
    ap.add_argument("--workers",    type=int,   default=4)
    ap.add_argument("--obj-thr",    type=float, default=0.25)
    ap.add_argument("--iou-thr",    type=float, default=0.5)
    ap.add_argument("--num-classes",type=int,   default=3)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-vis",     action="store_true", help="skip saving images")
    return ap.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    out    = Path(args.out_dir)
    vis_dir   = out / "vis";   vis_dir.mkdir(parents=True, exist_ok=True)
    plot_dir  = out / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ──
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone_size = ckpt.get("args", {}).get("backbone_size", "n")
    model = OGCDENet(nc=args.num_classes, backbone_size=backbone_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.ckpt}  backbone={backbone_size}  epoch={ckpt.get('epoch','?')}")

    # ── data ──
    ds = KITTIOGCDEDataset(
        args.kitti_root, args.split,
        img_size=args.img_size, augment=False,
    )
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=args.workers, collate_fn=collate_ogcde,
    )

    evaluator = OGCDEEvaluator(iou_thr=args.iou_thr, class_agnostic=False)
    map_eval  = MAPEvaluator(num_classes=args.num_classes, iou_thr=args.iou_thr)

    # ── per-sample collectors for plotting ──
    matched_records = []   # per matched pair: cls, dist_pred, dist_gt, ale
    # for PR: (score, img_id, box_xyxy, cls)
    all_preds_raw = []
    all_gts_raw   = {}     # (img_id, cls) -> gt_boxes_xyxy
    n_gt_cls      = np.zeros(args.num_classes, dtype=np.int64)
    tp_cls = np.zeros(args.num_classes, dtype=np.int64)
    fp_cls = np.zeros(args.num_classes, dtype=np.int64)
    fn_cls = np.zeros(args.num_classes, dtype=np.int64)

    total_vis = 0
    print(f"Running inference on {len(ds)} images …")

    for imgs_t, targets, meta in loader:
        m      = meta[0]
        img_id = m["image_id"]
        fx, fy, cx, cy = m["fx"], m["fy"], m["cx"], m["cy"]

        img_path = os.path.join(args.kitti_root, "image_2", f"{img_id}.png")
        bgr_img  = cv2.imread(img_path)
        if bgr_img is None:
            print(f"[skip] {img_id}"); continue

        det = infer(model, bgr_img, args.img_size, args.obj_thr, args.iou_thr,
                    device, args.num_classes)

        bmask = targets["batch_idx"].numpy() == 0
        gt = {
            "boxes":   unletterbox_boxes(targets["boxes"].numpy()[bmask], m["ratio"], m["pad"]),
            "labels":  targets["labels"].numpy()[bmask],
            "dist":    targets["dist"].numpy()[bmask],
            "depth":   targets["depth"].numpy()[bmask],
            "contact": unletterbox_points(targets["contact"].numpy()[bmask], m["ratio"], m["pad"]),
            "loc3d":   targets["loc3d"].numpy()[bmask],
        }

        # ── evaluators ──
        intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        evaluator.update(det, gt, intrinsics=intrinsics)
        map_eval.update(det, gt, img_id=img_id)

        # ── collect for plots ──
        g_xyxy = xywh_to_xyxy(gt["boxes"])
        p_xyxy = xywh_to_xyxy(det["boxes"])
        for ci in range(args.num_classes):
            mask = gt["labels"] == ci
            key  = (img_id, ci)
            all_gts_raw[key] = g_xyxy[mask]
            n_gt_cls[ci] += int(mask.sum())

        for i in range(len(det["boxes"])):
            all_preds_raw.append((
                float(det["scores"][i]), img_id, p_xyxy[i], int(det["classes"][i])
            ))

        if len(det["boxes"]) > 0 and len(gt["boxes"]) > 0:
            mp, mg, unm_p = greedy_match(
                p_xyxy, det["scores"], g_xyxy,
                iou_thr=args.iou_thr,
                pred_cls=det["classes"], gt_cls=gt["labels"],
            )
            matched_gt = set(mg.tolist())
            for pi, gi in zip(mp, mg):
                ci = int(gt["labels"][gi])
                dp = float(det["distance"][pi])
                dg = float(gt["dist"][gi])
                u, v = det["contact"][pi]
                Z    = float(det["depth"][pi])
                X_p  = (u - cx) * Z / fx
                Y_p  = (v - cy) * Z / fy
                X_g, Y_g, Z_g = gt["loc3d"][gi]
                ale = float(np.sqrt((X_p-X_g)**2 + (Y_p-Y_g)**2 + (Z-Z_g)**2))
                matched_records.append({"cls": ci, "dist_pred": dp, "dist_gt": dg, "ale": ale})
                tp_cls[ci] += 1
            for pi in unm_p:
                fp_cls[int(det["classes"][pi])] += 1
            for gi in range(len(gt["labels"])):
                if gi not in matched_gt:
                    fn_cls[int(gt["labels"][gi])] += 1
        else:
            for pi in range(len(det["boxes"])):
                fp_cls[int(det["classes"][pi])] += 1
            for gi in range(len(gt["labels"])):
                fn_cls[int(gt["labels"][gi])] += 1

        # ── save vis ──
        if not args.no_vis:
            vis = make_vis(bgr_img,
                           det["boxes"], det["classes"], det["scores"], det["distance"],
                           gt["boxes"], gt["labels"])
            cv2.imwrite(str(vis_dir / f"{img_id}.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            total_vis += 1

    # ── metrics ──
    result  = evaluator.compute()
    map_res = map_eval.compute()

    dm = result["distance"]; dz = result["depth"]
    print("\n" + "=" * 62)
    print(f"  Test images: {len(ds)}   Matched pairs: {result['n_matched']}")
    print("\n  [Detection]")
    for name, ap, ap3d in zip(CLASS_NAMES, map_res["AP_per_class"], map_res["AP3D_per_class"]):
        print(f"    AP   {name:<12s} = {ap:.4f}")
        print(f"    AP3D {name:<12s} = {ap3d:.4f}  (IoU≥0.5 + dist_ratio<1.25)")
    print(f"    mAP                 = {map_res['mAP']:.4f}")
    print(f"    mAP3D               = {map_res['mAP3D']:.4f}")
    print("\n  [Distance metrics (matched pairs)]")
    print(f"    AbsRel  = {dm['AbsRel']:.4f}")
    print(f"    RMSE    = {dm['RMSE']:.4f} m")
    print(f"    δ<1.25  = {dm['delta1']*100:.2f}%")
    print(f"    δ<1.25² = {dm['delta2']*100:.2f}%")
    print(f"    δ<1.25³ = {dm['delta3']*100:.2f}%")
    print("\n  [Depth (Z) metrics]")
    print(f"    AbsRel  = {dz['AbsRel']:.4f}")
    print(f"    RMSE    = {dz['RMSE']:.4f} m")
    print(f"    δ<1.25  = {dz['delta1']*100:.2f}%")
    print(f"    δ<1.25² = {dz['delta2']*100:.2f}%")
    print(f"    δ<1.25³ = {dz['delta3']*100:.2f}%")
    print(f"\n  DE  (distance MAE)          = {result['DE']:.4f} m")
    print(f"  CPE (contact point err)     = {result['CPE_px']:.4f} px")
    print(f"  ALE (3D loc error)          = {result['ALE']:.4f} m")
    print(f"\n  TP/FP/FN per class:")
    for ci, name in enumerate(CLASS_NAMES):
        print(f"    {name:<12s}  TP={tp_cls[ci]}  FP={fp_cls[ci]}  FN={fn_cls[ci]}")
    print("=" * 62)

    # ── save JSON ──
    summary = {
        "n_images": len(ds), "n_matched": result["n_matched"],
        "mAP": map_res["mAP"], "mAP3D": map_res["mAP3D"],
        "AP_per_class":   map_res["AP_per_class"],
        "AP3D_per_class": map_res["AP3D_per_class"],
        "distance": result["distance"], "depth": result["depth"],
        "DE": result["DE"], "CPE_px": result["CPE_px"], "ALE": result["ALE"],
        "TP": tp_cls.tolist(), "FP": fp_cls.tolist(), "FN": fn_cls.tolist(),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── plots ──
    print("\nGenerating plots …")
    plot_scatter(matched_records,   plot_dir / "dist_scatter.png")
    plot_error_hist(matched_records, plot_dir / "error_analysis.png")
    plot_dist_distribution(matched_records, plot_dir / "dist_distribution.png")
    plot_ale_vs_dist(matched_records, plot_dir / "ale_vs_dist.png")
    plot_pr_curves(all_preds_raw, all_gts_raw, n_gt_cls, args.iou_thr,
                   plot_dir / "pr_curves.png")
    plot_class_stats(matched_records,
                     tp_cls.tolist(), fp_cls.tolist(), fn_cls.tolist(),
                     map_res["AP_per_class"], map_res["AP3D_per_class"],
                     plot_dir / "class_stats.png")

    print(f"Plots saved → {plot_dir}/")
    if not args.no_vis:
        print(f"Vis images  → {vis_dir}/  ({total_vis} files)")
    print(f"Metrics JSON→ {out}/metrics.json")


if __name__ == "__main__":
    main()
