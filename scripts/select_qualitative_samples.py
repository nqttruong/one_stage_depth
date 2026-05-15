"""Select top-5 qualitative figure candidates per panel.

Panels:
  (a) Near Car       — GT 8-15m,  AbsRel < 5%,  occluded=0
  (b) Far Car        — GT > 50m,  AbsRel < 15%
  (c) Occluded Ped   — GT 5-20m,  occluded >= 1, AbsRel > 20%  (failure)
  (d) Cyclist        — any range, AbsRel < 10%

Output: figs/qualitative_candidates.json
  {
    "panel_a": {"candidates": [...top-5...], "chosen": <rank-1>},
    ...
  }

Usage:
    python scripts/select_qualitative_samples.py \\
        --ckpt runs/ogcde_lidar_p2/best.pt \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt \\
        --lidar-gt cache/lidar_gt_val.json \\
        --out figs/qualitative_candidates.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from ogcde.model import OGCDENet
from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde, parse_kitti_label
from ogcde.utils import decode_predictions, unletterbox_boxes, unletterbox_points
from ogcde.metrics import bbox_iou_xyxy, xywh_to_xyxy

CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

# Panel definitions: (class_id, gt_dist_range, absrel_range, occluded_range, rank_key)
# absrel_range = (min, max) — None means no bound
PANELS = {
    "panel_a": dict(
        title="(a) Near Car",
        class_id=0,
        dist_min=8.0, dist_max=15.0,
        absrel_min=0.02,   # want visible but small error (2-5%)
        absrel_max=0.05,
        occluded_max=0,
        want_failure=False,
        rank_by="absrel_asc",
    ),
    "panel_b": dict(
        title="(b) Far Car (>50m)",
        class_id=0,
        dist_min=50.0, dist_max=999.0,
        absrel_max=0.15,
        occluded_max=None,
        absrel_min=None,
        want_failure=False,
        rank_by="dist_desc",    # furthest first
    ),
    "panel_c": dict(
        title="(c) Occluded Pedestrian (failure)",
        class_id=1,
        dist_min=5.0, dist_max=20.0,
        absrel_min=0.20,        # WANT high error
        absrel_max=None,
        occluded_min=1,
        want_failure=True,
        rank_by="absrel_desc",  # worst error first
    ),
    "panel_d": dict(
        title="(d) Cyclist",
        class_id=2,
        dist_min=0.0, dist_max=999.0,
        absrel_max=0.10,
        occluded_max=None,
        absrel_min=None,
        want_failure=False,
        rank_by="absrel_asc",
    ),
}


def load_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       default="runs/ogcde_lidar_p2/best.pt")
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--split",      default="splits/distformer_val.txt")
    ap.add_argument("--lidar-gt",   default="cache/lidar_gt_val.json")
    ap.add_argument("--out",        default="figs/qualitative_candidates.json")
    ap.add_argument("--conf",       type=float, default=0.25)
    ap.add_argument("--nms-iou",    type=float, default=0.45)
    ap.add_argument("--top",        type=int,   default=5)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = load_args()
    device = torch.device(args.device)

    # ── model ────────────────────────────────────────────────────────────
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    bs    = ckpt.get("args", {}).get("backbone_size", "n")
    nc    = ckpt.get("args", {}).get("num_classes", 3)
    model = OGCDENet(nc=nc, backbone_size=bs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── lidar GT ─────────────────────────────────────────────────────────
    with open(args.lidar_gt) as f:
        lidar_db = json.load(f)

    # ── dataset ──────────────────────────────────────────────────────────
    ds = KITTIOGCDEDataset(
        args.kitti_root, args.split,
        img_size=640, augment=False, filter_difficult=False,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        num_workers=4, collate_fn=collate_ogcde)
    label_dir = os.path.join(args.kitti_root, "label_2")

    # ── collect detections ───────────────────────────────────────────────
    records = []   # list of dicts, one per matched (pred, GT) pair

    with torch.no_grad():
        for imgs, targets, meta in loader:
            imgs  = imgs.to(device, non_blocking=True)
            preds = model(imgs)
            dets  = decode_predictions(preds, nc,
                                       obj_thr=args.conf, iou_thr=args.nms_iou)

            for b, det in enumerate(dets):
                m      = meta[b]
                img_id = m["image_id"]
                bmask  = targets["batch_idx"].numpy() == b

                # GT in original coords
                gt_boxes_xywh = targets["boxes"].numpy()[bmask]
                gt_labels     = targets["labels"].numpy()[bmask]
                gt_obj_idxs   = targets["obj_idx"].numpy()[bmask]
                gt_boxes_xyxy = xywh_to_xyxy(
                    unletterbox_boxes(gt_boxes_xywh, m["ratio"], m["pad"]))

                # LiDAR distances keyed by obj_idx
                lidar_records = {r["obj_idx"]: r
                                 for r in lidar_db.get(str(img_id), [])}

                # Predictions in original coords
                pred_boxes_xywh = unletterbox_boxes(
                    det["boxes"].cpu().numpy(), m["ratio"], m["pad"])
                pred_contact = unletterbox_points(
                    det["contact"].cpu().numpy(), m["ratio"], m["pad"])
                pred_boxes_xyxy = xywh_to_xyxy(pred_boxes_xywh)
                pred_dist  = det["distance"].cpu().numpy()
                pred_depth = det["depth"].cpu().numpy()
                pred_cls   = det["classes"].cpu().numpy()
                pred_score = det["scores"].cpu().numpy()

                if len(pred_boxes_xyxy) == 0 or len(gt_boxes_xyxy) == 0:
                    continue

                ious = bbox_iou_xyxy(pred_boxes_xyxy, gt_boxes_xyxy)  # (P, G)

                # Read occlusion from label file
                raw_objs = parse_kitti_label(
                    os.path.join(label_dir, f"{img_id}.txt"))

                for pi in range(len(pred_boxes_xyxy)):
                    c = int(pred_cls[pi])
                    # match only same class
                    cls_mask = gt_labels == c
                    row = ious[pi].copy()
                    row[~cls_mask] = -1
                    if row.max() < 0.5:
                        continue
                    gi = int(row.argmax())
                    raw_idx = int(gt_obj_idxs[gi])

                    # LiDAR GT distance
                    lr = lidar_records.get(raw_idx, None)
                    if lr is None or lr.get("n_points", 0) == 0:
                        continue
                    gt_dist_lidar = lr["dist_lidar"]
                    absrel = abs(pred_dist[pi] - gt_dist_lidar) / gt_dist_lidar

                    # occlusion from raw label
                    occ = raw_objs[raw_idx]["occluded"] if raw_idx < len(raw_objs) else 0

                    records.append({
                        "img_id":   img_id,
                        "class_id": c,
                        "class_name": CLASS_NAMES.get(c, "?"),
                        "pred_dist":  float(pred_dist[pi]),
                        "pred_depth": float(pred_depth[pi]),
                        "gt_dist":    float(gt_dist_lidar),
                        "absrel":     float(absrel),
                        "occluded":   int(occ),
                        "score":      float(pred_score[pi]),
                        "pred_box_xywh": pred_boxes_xywh[pi].tolist(),
                        "pred_box_xyxy": pred_boxes_xyxy[pi].tolist(),
                        "pred_contact":  pred_contact[pi].tolist(),
                        "gt_box_xyxy":   gt_boxes_xyxy[gi].tolist(),
                        "ratio":  m["ratio"],
                        "pad":    list(m["pad"]),
                    })

    print(f"Total matched pairs collected: {len(records)}")

    # ── filter & rank per panel ──────────────────────────────────────────
    result = {}
    all_ok = True

    for panel_id, cfg in PANELS.items():
        candidates = []
        for r in records:
            if r["class_id"] != cfg["class_id"]:
                continue
            if r["gt_dist"] < cfg.get("dist_min", 0):
                continue
            if r["gt_dist"] > cfg.get("dist_max", 999):
                continue
            if cfg.get("absrel_max") is not None and r["absrel"] > cfg["absrel_max"]:
                continue
            if cfg.get("absrel_min") is not None and r["absrel"] < cfg["absrel_min"]:
                continue
            if cfg.get("occluded_max") is not None and r["occluded"] > cfg["occluded_max"]:
                continue
            if cfg.get("occluded_min") is not None and r["occluded"] < cfg["occluded_min"]:
                continue
            candidates.append(r)

        rank_by = cfg.get("rank_by", "absrel_asc")
        if rank_by == "absrel_asc":
            candidates.sort(key=lambda x: x["absrel"])
        elif rank_by == "absrel_desc":
            candidates.sort(key=lambda x: -x["absrel"])
        elif rank_by == "dist_desc":
            candidates.sort(key=lambda x: -x["gt_dist"])

        top = candidates[:args.top]
        print(f"\n{cfg['title']}: {len(candidates)} candidates, showing top {len(top)}")
        for i, c in enumerate(top):
            print(f"  #{i+1} img={c['img_id']}  gt={c['gt_dist']:.1f}m  "
                  f"pred={c['pred_dist']:.1f}m  AbsRel={c['absrel']*100:.1f}%  "
                  f"occ={c['occluded']}")

        if cfg.get("want_failure") and len(top) == 0:
            print(f"\n⚠️  STOP: No panel (c) candidate found with AbsRel > 20%!")
            print("Options:")
            print("  1. Lower absrel_min threshold in PANELS config")
            print("  2. Use a missed detection (FN) instead")
            all_ok = False

        result[panel_id] = {
            "title":      cfg["title"],
            "candidates": top,
            "chosen":     top[0] if top else None,
        }

    # ── save ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {args.out}")

    if not all_ok:
        print("\n❌ Some panels have no candidates. Edit thresholds or JSON manually.")
        sys.exit(1)
    else:
        print("\n✅ All 4 panels have candidates. Edit JSON to swap chosen if needed.")


if __name__ == "__main__":
    main()
