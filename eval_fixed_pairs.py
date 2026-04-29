"""Evaluate multiple OGCDE checkpoints on a FIXED set of GT objects.

Problem: DE/AbsRel across runs is confounded by recall differences — a model
that detects more (harder) objects looks worse even if its distance estimation
is better. This script finds the INTERSECTION of GT objects matched by all
models and re-computes metrics on that common set only.

Usage:
    python eval_fixed_pairs.py \
        --ckpts runs/ogcde_v1/best.pt runs/ogcde_v2/best.pt \
                runs/ogcde_v3/best.pt runs/ogcde_v4/best.pt \
        --names v1 v2 v3 v4 \
        --kitti-root /data/kitti/training \
        --split splits/kitti_val.txt
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ogcde.model import OGCDENet
from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde
from ogcde.utils import decode_predictions, unletterbox_boxes, unletterbox_points
from ogcde.metrics import (
    xywh_to_xyxy, greedy_match, depth_metrics, distance_error, contact_point_error
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None,
                    help="Display names for each checkpoint (default: basename).")
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--obj-thr", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def run_one_model(ckpt_path, ds, loader, args, device):
    """Run inference and return per-image matched GT records.

    Returns dict: {(img_id, gt_idx): {"dist_pred", "depth_pred", "cp_pred"}}
    """
    model = OGCDENet(nc=args.num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    records = {}  # (img_id, gt_idx) -> prediction values

    with torch.no_grad():
        for imgs, targets, meta in loader:
            imgs = imgs.to(device, non_blocking=True)
            preds = model(imgs)
            dets = decode_predictions(
                preds, args.num_classes,
                obj_thr=args.obj_thr, iou_thr=args.iou_thr,
            )

            gt_boxes_all  = targets["boxes"].numpy()
            gt_labels_all = targets["labels"].numpy()
            gt_dist_all   = targets["dist"].numpy()
            gt_depth_all  = targets["depth"].numpy()
            gt_contact_all = targets["contact"].numpy()
            batch_idx_all = targets["batch_idx"].numpy()

            for b, det in enumerate(dets):
                m = meta[b]
                img_id = m["image_id"]

                bmask = batch_idx_all == b
                gt_idxs = np.where(bmask)[0]  # positions in flat GT arrays
                # local gt index within image = 0..n_gt-1
                local_gt_indices = np.arange(int(bmask.sum()))

                gt_boxes   = unletterbox_boxes(gt_boxes_all[bmask], m["ratio"], m["pad"])
                gt_labels  = gt_labels_all[bmask]
                gt_dist    = gt_dist_all[bmask]
                gt_depth   = gt_depth_all[bmask]
                gt_contact = unletterbox_points(gt_contact_all[bmask], m["ratio"], m["pad"])

                pred_boxes   = unletterbox_boxes(det["boxes"].cpu().numpy(), m["ratio"], m["pad"])
                pred_contact = unletterbox_points(det["contact"].cpu().numpy(), m["ratio"], m["pad"])
                pred_scores  = det["scores"].cpu().numpy()
                pred_classes = det["classes"].cpu().numpy()
                pred_dist    = det["distance"].cpu().numpy()
                pred_depth   = det["depth"].cpu().numpy()

                if len(pred_boxes) == 0:
                    continue

                mp, mg, _ = greedy_match(
                    xywh_to_xyxy(pred_boxes), pred_scores,
                    xywh_to_xyxy(gt_boxes), iou_thr=args.iou_thr,
                    pred_cls=pred_classes, gt_cls=gt_labels,
                )
                for pi, gi in zip(mp, mg):
                    key = (img_id, int(local_gt_indices[gi]))
                    records[key] = {
                        "dist_pred":  float(pred_dist[pi]),
                        "dist_gt":    float(gt_dist[gi]),
                        "depth_pred": float(pred_depth[pi]),
                        "depth_gt":   float(gt_depth[gi]),
                        "cp_pred":    pred_contact[pi].tolist(),
                        "cp_gt":      gt_contact[gi].tolist(),
                    }

    return records


def compute_metrics_on_keys(records, keys):
    """Compute DE/AbsRel/δ1 for a subset of keys."""
    dist_pred, dist_gt = [], []
    depth_pred, depth_gt = [], []
    cp_pred, cp_gt = [], []
    for k in keys:
        r = records[k]
        dist_pred.append(r["dist_pred"])
        dist_gt.append(r["dist_gt"])
        depth_pred.append(r["depth_pred"])
        depth_gt.append(r["depth_gt"])
        cp_pred.append(r["cp_pred"])
        cp_gt.append(r["cp_gt"])

    dist_m  = depth_metrics(np.array(dist_pred),  np.array(dist_gt))
    depth_m = depth_metrics(np.array(depth_pred), np.array(depth_gt))
    de  = distance_error(np.array(dist_pred), np.array(dist_gt))
    cpe = contact_point_error(np.array(cp_pred), np.array(cp_gt))
    return dist_m, depth_m, de, cpe


def main():
    args = parse_args()
    device = torch.device(args.device)
    names = args.names or [p.split("/")[-2] for p in args.ckpts]

    ds = KITTIOGCDEDataset(
        args.kitti_root, args.split,
        img_size=args.img_size, augment=False,
    )
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=collate_ogcde,
    )

    # Run each model and collect matched GT keys
    all_records = {}
    all_keys = []
    for name, ckpt_path in zip(names, args.ckpts):
        print(f"Running {name} ({ckpt_path}) ...")
        rec = run_one_model(ckpt_path, ds, loader, args, device)
        all_records[name] = rec
        all_keys.append(set(rec.keys()))
        print(f"  matched {len(rec)} pairs")

    # Intersection: GT objects detected by ALL models
    common_keys = all_keys[0].copy()
    for ks in all_keys[1:]:
        common_keys &= ks
    print(f"\nIntersection: {len(common_keys)} pairs matched by all {len(args.ckpts)} models\n")

    # Print results on full set vs intersection
    header = f"{'Model':<10} {'All pairs':>10} {'AbsRel':>8} {'DE':>8} {'δ1':>8} {'CPE':>8}"
    sep    = "-" * len(header)
    print("── Full set (each model's own detections) ──")
    print(header)
    print(sep)
    for name in names:
        rec = all_records[name]
        dist_m, _, de, cpe = compute_metrics_on_keys(rec, rec.keys())
        print(f"{name:<10} {len(rec):>10} {dist_m['AbsRel']:>8.4f} {de:>8.4f}"
              f" {dist_m['delta1']:>8.4f} {cpe:>8.4f}")

    print(f"\n── Fixed intersection ({len(common_keys)} pairs) ──")
    print(header)
    print(sep)
    for name in names:
        rec = all_records[name]
        dist_m, _, de, cpe = compute_metrics_on_keys(rec, common_keys)
        print(f"{name:<10} {len(common_keys):>10} {dist_m['AbsRel']:>8.4f} {de:>8.4f}"
              f" {dist_m['delta1']:>8.4f} {cpe:>8.4f}")

    print()


if __name__ == "__main__":
    main()
