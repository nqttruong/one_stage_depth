"""Run OGCDE evaluation on a KITTI split.

Computes:
    - AbsRel, RMSE, RMSE_log, δ1/δ2/δ3 on distance (and on depth)
    - DE  (distance error)
    - CPE (contact point pixel error)

Usage:
    python evaluate_kitti.py --ckpt runs/ogcde/best.pt \
        --kitti-root /data/kitti/training \
        --split splits/kitti_val.txt
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ogcde.model import OGCDENet
from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde
from ogcde.utils import decode_predictions, unletterbox_boxes, unletterbox_points
from ogcde.metrics import OGCDEEvaluator


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--obj-thr", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-json", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # -------- model ---------
    model = OGCDENet(nc=args.num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # -------- data ----------
    ds = KITTIOGCDEDataset(
        args.kitti_root, args.split,
        img_size=args.img_size, augment=False,
    )
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=collate_ogcde,
    )

    evaluator = OGCDEEvaluator(iou_thr=args.iou_thr, class_agnostic=False)

    with torch.no_grad():
        for imgs, targets, meta in loader:
            imgs = imgs.to(device, non_blocking=True)
            preds = model(imgs)
            dets = decode_predictions(
                preds, args.num_classes,
                obj_thr=args.obj_thr, iou_thr=args.iou_thr,
            )

            for b, det in enumerate(dets):
                m = meta[b]
                # map predictions back to original image coordinates
                pred_boxes = unletterbox_boxes(det["boxes"].cpu().numpy(), m["ratio"], m["pad"])
                pred_contact = unletterbox_points(det["contact"].cpu().numpy(), m["ratio"], m["pad"])

                # build GT in original coords
                gt_boxes = targets["boxes"].numpy()
                gt_labels = targets["labels"].numpy()
                gt_dist = targets["dist"].numpy()
                gt_depth = targets["depth"].numpy()
                gt_contact = targets["contact"].numpy()

                # select GT entries that belong to this image
                bmask = (targets["batch_idx"].numpy() == b)
                gt = {
                    "boxes": unletterbox_boxes(gt_boxes[bmask], m["ratio"], m["pad"]),
                    "labels": gt_labels[bmask],
                    "dist": gt_dist[bmask],
                    "depth": gt_depth[bmask],
                    "contact": unletterbox_points(gt_contact[bmask], m["ratio"], m["pad"]),
                }
                pred = {
                    "boxes": pred_boxes,
                    "scores": det["scores"].cpu().numpy(),
                    "classes": det["classes"].cpu().numpy(),
                    "depth": det["depth"].cpu().numpy(),
                    "distance": det["distance"].cpu().numpy(),
                    "contact": pred_contact,
                }
                evaluator.update(pred, gt)

    result = evaluator.compute()

    print("=" * 60)
    print(f"Matched pairs: {result['n_matched']}")
    print("\n[Distance metrics]")
    for k, v in result["distance"].items():
        print(f"  {k:10s} = {v:.4f}" if isinstance(v, float) else f"  {k:10s} = {v}")
    print("\n[Depth metrics]")
    for k, v in result["depth"].items():
        print(f"  {k:10s} = {v:.4f}" if isinstance(v, float) else f"  {k:10s} = {v}")
    print(f"\nDE (mean |dist_pred - dist_gt|) = {result['DE']:.4f} m")
    print(f"CPE (contact point pixel err)    = {result['CPE_px']:.4f} px")
    print("=" * 60)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.save_json}")


if __name__ == "__main__":
    main()
