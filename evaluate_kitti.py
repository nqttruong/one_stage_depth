"""Run OGCDE evaluation on a KITTI split.

Modes (--mode):
  iou    — default: IoU≥0.5 matched pairs, full detection + distance eval
  oracle — GT-box oracle: query model at GT center cell (no detection matching),
           mirrors DistFormer / two-stage baselines that receive GT boxes
  both   — run both modes and print side-by-side

GT distance source (--lidar-gt):
  By default uses annotation-based distance from KITTI label loc field.
  With --lidar-gt <json>, replaces GT distance with LiDAR-based values
  produced by prepare_lidar_gt.py (Zhu et al. / DistFormer methodology).

Usage:
    python evaluate_kitti.py --ckpt runs/ogcde/best.pt \\
        --kitti-root /data/kitti/training \\
        --split splits/kitti_val.txt

    # oracle mode with LiDAR GT (fair comparison with DistFormer):
    python evaluate_kitti.py --ckpt runs/ogcde/best.pt \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt \\
        --mode oracle --lidar-gt cache/lidar_gt_val.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from ogcde.model import OGCDENet
from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde
from ogcde.utils import decode_predictions, unletterbox_boxes, unletterbox_points
from ogcde.metrics import OGCDEEvaluator, MAPEvaluator
from ogcde.oracle import oracle_query, OracleEvaluator

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",         required=True)
    ap.add_argument("--kitti-root",   required=True)
    ap.add_argument("--split",        required=True)
    ap.add_argument("--img-size",     type=int,   default=640)
    ap.add_argument("--batch",        type=int,   default=16)
    ap.add_argument("--workers",      type=int,   default=4)
    ap.add_argument("--obj-thr",      type=float, default=0.25)
    ap.add_argument("--iou-thr",      type=float, default=0.5)
    ap.add_argument("--num-classes",  type=int,   default=3)
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-json",    default=None)
    ap.add_argument("--lidar-gt",     default=None,
                    help="JSON from prepare_lidar_gt.py — replaces annotation GT distance")
    ap.add_argument(
        "--mode", choices=["iou", "oracle", "both"], default="iou",
        help="iou: IoU≥thr matched pairs (default) | "
             "oracle: query at GT center cell | "
             "both: run and show both",
    )
    return ap.parse_args()


def _print_depth_block(m):
    print(f"  δ<1.25  = {m['delta1']:.4f}  AbsRel  = {m['AbsRel']:.4f}  SqRel   = {m['SqRel']:.4f}")
    print(f"  RMSE    = {m['RMSE']:.4f} m  RMSElog = {m['RMSE_log']:.4f}")
    print(f"  δ<1.25² = {m['delta2']:.4f}  δ<1.25³ = {m['delta3']:.4f}  n = {m['n']}")


def print_iou_result(result, map_res):
    print("=" * 60)
    print(f"[IOU MODE]  Matched pairs: {result['n_matched']}")
    print("\n[Detection]")
    for name, ap, ap3d in zip(CLASS_NAMES, map_res["AP_per_class"], map_res["AP3D_per_class"]):
        print(f"  AP   {name:<12s} = {ap:.4f}")
        print(f"  AP3D {name:<12s} = {ap3d:.4f}  (IoU≥0.5 + dist_ratio<1.25)")
    print(f"  mAP                = {map_res['mAP']:.4f}")
    print(f"  mAP3D              = {map_res['mAP3D']:.4f}")

    print("\n[Distance — all classes]")
    _print_depth_block(result["distance"])
    print(f"  DE  = {result['DE']:.4f} m   ALE = {result['ALE']:.4f} m")

    print("\n[Distance — per class]")
    for name, pc in zip(CLASS_NAMES, result["per_class"]):
        print(f"  {name}:")
        _print_depth_block(pc["distance"])
        print(f"    DE = {pc['DE']:.4f} m   ALE = {pc['ALE']:.4f} m")

    print("\n[Depth (Z) — all classes]")
    _print_depth_block(result["depth"])

    print("\n[Depth (Z) — per class]")
    for name, pc in zip(CLASS_NAMES, result["per_class"]):
        print(f"  {name}:")
        _print_depth_block(pc["depth"])

    print(f"\nCPE = {result['CPE_px']:.4f} px")
    print("=" * 60)


def print_oracle_result(result):
    print("=" * 60)
    print(f"[ORACLE MODE]  GT objects queried: {result['n_objects']}")

    print("\n[Distance — all classes]")
    _print_depth_block(result["distance"])
    print(f"  DE  = {result['DE']:.4f} m   ALE = {result['ALE']:.4f} m")

    print("\n[Distance — per class]")
    for name, pc in zip(CLASS_NAMES, result["per_class"]):
        print(f"  {name}:")
        _print_depth_block(pc["distance"])
        print(f"    DE = {pc['DE']:.4f} m   ALE = {pc['ALE']:.4f} m")

    print("\n[Depth (Z) — all classes]")
    _print_depth_block(result["depth"])

    print("\n[Depth (Z) — per class]")
    for name, pc in zip(CLASS_NAMES, result["per_class"]):
        print(f"  {name}:")
        _print_depth_block(pc["depth"])

    print(f"\nCPE = {result['CPE_px']:.4f} px")
    print("=" * 60)


def main():
    args = parse_args()
    device = torch.device(args.device)
    run_iou    = args.mode in ("iou", "both")
    run_oracle = args.mode in ("oracle", "both")

    # -------- LiDAR GT (optional) ----
    lidar_gt_db = None
    if args.lidar_gt:
        with open(args.lidar_gt) as f:
            lidar_gt_db = json.load(f)
        print(f"[lidar-gt] Loaded {len(lidar_gt_db)} images from {args.lidar_gt}")

    # -------- model ---------
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone_size = ckpt.get("args", {}).get("backbone_size", "n")
    model = OGCDENet(nc=args.num_classes, backbone_size=backbone_size).to(device)
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

    # -------- evaluators ----
    evaluator  = OGCDEEvaluator(iou_thr=args.iou_thr, class_agnostic=False) if run_iou else None
    map_eval   = MAPEvaluator(num_classes=args.num_classes, iou_thr=args.iou_thr)          if run_iou else None
    ora_eval   = OracleEvaluator(num_classes=args.num_classes)                              if run_oracle else None

    with torch.no_grad():
        for imgs, targets, meta in loader:
            imgs  = imgs.to(device, non_blocking=True)
            preds = model(imgs)           # raw grid outputs, list of 3 tensors

            # decode predictions for IoU mode (NMS applied)
            dets = decode_predictions(
                preds, args.num_classes,
                obj_thr=args.obj_thr, iou_thr=args.iou_thr,
            ) if run_iou else [None] * imgs.shape[0]

            for b in range(imgs.shape[0]):
                m = meta[b]
                bmask = targets["batch_idx"].numpy() == b

                gt_dist_annot = targets["dist"].numpy()[bmask]

                # optionally replace GT distance with LiDAR-based values
                if lidar_gt_db is not None:
                    img_id = m["image_id"]
                    records = lidar_gt_db.get(str(img_id), [])
                    gt_obj_idx = targets.get("obj_idx", None)
                    if gt_obj_idx is not None:
                        idx_arr = gt_obj_idx.numpy()[bmask]
                        rec_by_idx = {r["obj_idx"]: r["dist_lidar"] for r in records}
                        gt_dist_use = np.array([
                            rec_by_idx.get(int(i), gt_dist_annot[k])
                            for k, i in enumerate(idx_arr)
                        ])
                    else:
                        # fallback: align by order within image
                        lidar_dists = [r["dist_lidar"] for r in records]
                        n = len(gt_dist_annot)
                        gt_dist_use = np.array(
                            lidar_dists[:n] if len(lidar_dists) >= n
                            else lidar_dists + gt_dist_annot[len(lidar_dists):].tolist()
                        )
                else:
                    gt_dist_use = gt_dist_annot

                gt = {
                    "boxes":   unletterbox_boxes(
                                   targets["boxes"].numpy()[bmask], m["ratio"], m["pad"]),
                    "labels":  targets["labels"].numpy()[bmask],
                    "dist":    gt_dist_use,
                    "depth":   targets["depth"].numpy()[bmask],
                    "contact": unletterbox_points(
                                   targets["contact"].numpy()[bmask], m["ratio"], m["pad"]),
                    "loc3d":   targets["loc3d"].numpy()[bmask],
                }
                intrinsics = {"fx": m["fx"], "fy": m["fy"], "cx": m["cx"], "cy": m["cy"]}

                # --- IoU mode ---
                if run_iou:
                    det = dets[b]
                    pred = {
                        "boxes":    unletterbox_boxes(det["boxes"].cpu().numpy(), m["ratio"], m["pad"]),
                        "scores":   det["scores"].cpu().numpy(),
                        "classes":  det["classes"].cpu().numpy(),
                        "depth":    det["depth"].cpu().numpy(),
                        "distance": det["distance"].cpu().numpy(),
                        "contact":  unletterbox_points(det["contact"].cpu().numpy(), m["ratio"], m["pad"]),
                    }
                    evaluator.update(pred, gt, intrinsics=intrinsics)
                    map_eval.update(pred, gt, img_id=m["image_id"])

                # --- Oracle mode ---
                if run_oracle:
                    ora_pred = oracle_query(
                        preds, b, gt["boxes"], m,
                        nc=args.num_classes,
                        strides=(8, 16, 32),
                    )
                    ora_eval.update(ora_pred, gt, intrinsics=intrinsics)

    # -------- results -------
    iou_result = ora_result = None
    if run_iou:
        map_res    = map_eval.compute()
        iou_result = evaluator.compute(num_classes=args.num_classes)
        iou_result["mAP"] = map_res
        print_iou_result(iou_result, map_res)

    if run_oracle:
        ora_result = ora_eval.compute()
        print_oracle_result(ora_result)

    if args.save_json:
        out = {}
        if iou_result: out["iou"]    = iou_result
        if ora_result: out["oracle"] = ora_result
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved to {args.save_json}")


if __name__ == "__main__":
    main()
