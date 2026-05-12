"""Full detection evaluation for OGCDE.

Produces:
  1. AP@0.5 and AP@0.5:0.95 per class
  2. KITTI Easy / Moderate / Hard AP per class
  3. Recall @ distance bins: 0-20 / 20-40 / 40-60 / >60 m

KITTI difficulty (per official devkit):
  Easy:     bbox height >= 40px, occlusion <= 0, truncation <= 0.15
  Moderate: bbox height >= 25px, occlusion <= 1, truncation <= 0.30
  Hard:     bbox height >= 25px, occlusion <= 2, truncation <= 0.50

Usage:
    python eval_detection.py \\
        --ckpt runs/ogcde_lidar_p2/best.pt \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt \\
        --conf 0.001 --nms-iou 0.5 \\
        --save-json runs/eval/detection_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ogcde.model import OGCDENet
from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde, parse_kitti_label
from ogcde.utils import decode_predictions, unletterbox_boxes
from ogcde.metrics import bbox_iou_xyxy, xywh_to_xyxy, compute_ap

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
CLASS_MAP   = {"Car": 0, "Van": 0, "Truck": 0,
               "Pedestrian": 1, "Person_sitting": 1,
               "Cyclist": 2}

DIST_BINS   = [(0, 20), (20, 40), (40, 60), (60, 999)]
DIST_LABELS = ["0-20 m", "20-40 m", "40-60 m", ">60 m"]

# Official KITTI IoU thresholds per class (for 2D bbox eval)
# Car=0.7, Pedestrian=0.5, Cyclist=0.5  (KITTI devkit standard)
KITTI_IOU_THR = {0: 0.7, 1: 0.5, 2: 0.5}

# ─── KITTI difficulty ────────────────────────────────────────────────────────

def kitti_difficulty(obj: dict) -> int | None:
    """Return 0=Easy, 1=Moderate, 2=Hard, None=ignored."""
    h = obj["bbox"][3] - obj["bbox"][1]           # pixel height
    occ  = obj["occluded"]
    trun = obj["truncated"]
    if h >= 40 and occ <= 0 and trun <= 0.15:
        return 0   # Easy
    if h >= 25 and occ <= 1 and trun <= 0.30:
        return 1   # Moderate
    if h >= 25 and occ <= 2 and trun <= 0.50:
        return 2   # Hard
    return None    # too small / too occluded → ignored


def load_gt_with_difficulty(label_dir: str, img_id: str):
    """Return (gt_list, dontcare_boxes).

    gt_list      : valid GT objects with difficulty + dist_m
    dontcare_boxes: (N,4) xyxy boxes for DontCare regions (used to neutralise FP)
    """
    path = os.path.join(label_dir, f"{img_id}.txt")
    out, dc_boxes = [], []
    for obj in parse_kitti_label(path):
        if obj["type"] == "DontCare":
            l, t, r, b = obj["bbox"]
            dc_boxes.append([l, t, r, b])
            continue
        if obj["type"] not in CLASS_MAP:
            continue
        diff = kitti_difficulty(obj)
        x, y, z = obj["loc"]
        dist_m = math.sqrt(x*x + y*y + z*z)
        out.append({
            **obj,
            "class_id":  CLASS_MAP[obj["type"]],
            "difficulty": diff,
            "dist_m":    dist_m,
        })
    return out, np.array(dc_boxes, dtype=np.float64).reshape(-1, 4)


# ─── Prediction collector ────────────────────────────────────────────────────

class DetectionCollector:
    """Collect raw predictions and GT for post-hoc AP computation."""

    def __init__(self, num_classes=3):
        self.nc = num_classes
        # per-class: list of (score, img_id, box_xyxy)
        self._preds: list[list] = [[] for _ in range(num_classes)]
        # per-(img_id, class): GT info list
        self._gts:  dict = {}

    def update(self, pred_boxes_xyxy, pred_scores, pred_classes,
               gt_list, img_id, dontcare_boxes=None):
        """
        pred_boxes_xyxy : (N,4) in original image coords
        pred_scores     : (N,)
        pred_classes    : (N,) int
        gt_list         : GT objects from load_gt_with_difficulty
        img_id          : str
        dontcare_boxes  : (M,4) xyxy — predictions overlapping these are ignored
        """
        dc = dontcare_boxes if dontcare_boxes is not None else np.zeros((0, 4))

        for i in range(len(pred_boxes_xyxy)):
            c = int(pred_classes[i])
            if not (0 <= c < self.nc):
                continue
            # Ignore prediction if it overlaps a DontCare region (IoU ≥ 0.5)
            if len(dc) > 0:
                dc_iou = bbox_iou_xyxy(pred_boxes_xyxy[i:i+1], dc)[0]
                if dc_iou.max() >= 0.5:
                    continue   # skip — don't add as FP or TP
            self._preds[c].append((
                float(pred_scores[i]),
                img_id,
                pred_boxes_xyxy[i],
            ))

        # store GT
        for c in range(self.nc):
            key = (img_id, c)
            self._gts[key] = [g for g in gt_list if g["class_id"] == c]

    # ── AP at a single IoU threshold, optionally filtered by max difficulty ──

    def _ap_class(self, c: int, iou_thr: float,
                  max_difficulty: int | None = None) -> float:
        """
        max_difficulty: 0=Easy only, 1=Mod&Easy, 2=Hard&Mod&Easy, None=all
        """
        preds_c = sorted(self._preds[c], key=lambda x: -x[0])
        if not preds_c:
            return 0.0

        # count total valid GT for this class / difficulty
        n_gt = 0
        for key, gts in self._gts.items():
            if key[1] != c:
                continue
            for g in gts:
                if max_difficulty is None:
                    n_gt += 1
                elif g["difficulty"] is not None and g["difficulty"] <= max_difficulty:
                    n_gt += 1

        if n_gt == 0:
            return float("nan")

        matched: dict = {}   # key -> bool array
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))

        for i, (score, img_id, pbox) in enumerate(preds_c):
            key = (img_id, c)
            gts_here = self._gts.get(key, [])

            if not gts_here:
                fp[i] = 1
                continue

            if key not in matched:
                matched[key] = np.zeros(len(gts_here), dtype=bool)

            # build GT boxes array (only valid difficulty)
            gt_boxes = []
            gt_valid = []
            for gi, g in enumerate(gts_here):
                l, t, r, b = g["bbox"]
                gt_boxes.append([l, t, r, b])
                if max_difficulty is None:
                    gt_valid.append(True)
                else:
                    gt_valid.append(
                        g["difficulty"] is not None
                        and g["difficulty"] <= max_difficulty
                    )

            gt_boxes = np.array(gt_boxes, dtype=np.float64)
            gt_valid = np.array(gt_valid)

            ious = bbox_iou_xyxy(pbox[None], gt_boxes)[0]
            # mask already matched and ignored GT
            ious[matched[key]] = -1
            ious[~gt_valid]    = -1

            best = int(np.argmax(ious))
            if ious[best] >= iou_thr and gt_valid[best]:
                if not matched[key][best]:
                    tp[i] = 1
                    matched[key][best] = True
                else:
                    fp[i] = 1
            else:
                # check if it overlaps an ignored GT → don't penalise
                ious_all = bbox_iou_xyxy(pbox[None], gt_boxes)[0]
                ious_all[matched[key]] = -1
                ignored_best = int(np.argmax(ious_all * (~gt_valid)))
                if ious_all[ignored_best] >= iou_thr and not gt_valid[ignored_best]:
                    tp[i] = 0; fp[i] = 0   # neutral — ignored region
                else:
                    fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec  = cum_tp / (n_gt + 1e-9)
        prec = cum_tp / (cum_tp + cum_fp + 1e-9)
        return compute_ap(rec, prec)

    def compute_ap50(self):
        """AP@0.5 uniform (for COCO-style reporting)."""
        return [self._ap_class(c, 0.5) for c in range(self.nc)]

    def compute_ap50_kitti_iou(self):
        """AP at official KITTI IoU per class: Car@0.7, Ped@0.5, Cyc@0.5."""
        return [self._ap_class(c, KITTI_IOU_THR.get(c, 0.5))
                for c in range(self.nc)]

    def compute_ap5095(self):
        """COCO mAP: average over IoU thresholds 0.50:0.05:0.95."""
        thrs = np.arange(0.50, 1.00, 0.05)
        per_class = []
        for c in range(self.nc):
            aps = [self._ap_class(c, t) for t in thrs]
            valid = [a for a in aps if not math.isnan(a)]
            per_class.append(float(np.mean(valid)) if valid else float("nan"))
        return per_class

    def compute_kitti(self):
        """KITTI Easy/Moderate/Hard AP using official per-class IoU threshold."""
        levels = {"Easy": 0, "Moderate": 1, "Hard": 2}
        result = {}
        for name, max_diff in levels.items():
            result[name] = [
                self._ap_class(c, KITTI_IOU_THR.get(c, 0.5), max_diff)
                for c in range(self.nc)
            ]
        return result

    # ── Recall @ distance bins ───────────────────────────────────────────────

    def compute_recall_by_distance(self, iou_thr=0.5):
        """
        Returns dict: class -> [{bin_label, tp, gt, recall}, ...]
        """
        result = {}
        for c in range(self.nc):
            preds_c = sorted(self._preds[c], key=lambda x: -x[0])
            matched: dict = {}

            # first pass: greedy match (score-sorted)
            tp_pairs = set()  # (img_id, gt_idx)
            for score, img_id, pbox in preds_c:
                key = (img_id, c)
                gts_here = self._gts.get(key, [])
                if not gts_here:
                    continue
                if key not in matched:
                    matched[key] = np.zeros(len(gts_here), dtype=bool)

                gt_boxes = np.array([g["bbox"] for g in gts_here])
                ious = bbox_iou_xyxy(pbox[None], gt_boxes)[0]
                ious[matched[key]] = -1
                best = int(np.argmax(ious))
                if ious[best] >= iou_thr:
                    matched[key][best] = True
                    tp_pairs.add((img_id, c, best))

            # bin GT by distance
            bins = {i: {"tp": 0, "gt": 0} for i in range(len(DIST_BINS))}
            for key, gts_here in self._gts.items():
                img_id, klass = key
                if klass != c:
                    continue
                for gi, g in enumerate(gts_here):
                    d = g["dist_m"]
                    for bi, (lo, hi) in enumerate(DIST_BINS):
                        if lo <= d < hi:
                            bins[bi]["gt"] += 1
                            if (img_id, c, gi) in tp_pairs:
                                bins[bi]["tp"] += 1
                            break

            result[CLASS_NAMES[c]] = [
                {
                    "bin":    DIST_LABELS[bi],
                    "tp":     bins[bi]["tp"],
                    "gt":     bins[bi]["gt"],
                    "recall": bins[bi]["tp"] / bins[bi]["gt"] if bins[bi]["gt"] > 0 else float("nan"),
                }
                for bi in range(len(DIST_BINS))
            ]
        return result


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",        required=True)
    ap.add_argument("--kitti-root",  required=True)
    ap.add_argument("--split",       required=True)
    ap.add_argument("--img-size",    type=int,   default=640)
    ap.add_argument("--batch",       type=int,   default=16)
    ap.add_argument("--workers",     type=int,   default=4)
    ap.add_argument("--conf",        type=float, default=0.001)
    ap.add_argument("--nms-iou",     type=float, default=0.5)
    ap.add_argument("--num-classes", type=int,   default=3)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-json",   default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    label_dir = os.path.join(args.kitti_root, "label_2")

    # model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    bs   = ckpt.get("args", {}).get("backbone_size", "n")
    model = OGCDENet(nc=args.num_classes, backbone_size=bs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # dataset — filter_difficult=False to keep all GT for difficulty eval
    ds = KITTIOGCDEDataset(
        args.kitti_root, args.split,
        img_size=args.img_size, augment=False,
        filter_difficult=False,
    )
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=collate_ogcde,
    )

    collector = DetectionCollector(num_classes=args.num_classes)

    with torch.no_grad():
        for imgs, targets, meta in loader:
            imgs = imgs.to(device, non_blocking=True)
            preds = model(imgs)
            dets  = decode_predictions(
                preds, args.num_classes,
                obj_thr=args.conf, iou_thr=args.nms_iou,
            )

            for b, det in enumerate(dets):
                m     = meta[b]
                bmask = targets["batch_idx"].numpy() == b

                pred_boxes = unletterbox_boxes(
                    det["boxes"].cpu().numpy(), m["ratio"], m["pad"])
                pred_xyxy  = xywh_to_xyxy(pred_boxes)

                gt_list, dc_boxes = load_gt_with_difficulty(
                    label_dir, m["image_id"])

                collector.update(
                    pred_xyxy,
                    det["scores"].cpu().numpy(),
                    det["classes"].cpu().numpy(),
                    gt_list,
                    m["image_id"],
                    dontcare_boxes=dc_boxes,
                )

    # ── compute all metrics ──────────────────────────────────────────────────
    ap50     = collector.compute_ap50()
    ap50_kit = collector.compute_ap50_kitti_iou()
    ap5095   = collector.compute_ap5095()
    kitti_ap = collector.compute_kitti()
    recall_d = collector.compute_recall_by_distance(iou_thr=0.5)

    # ── print ────────────────────────────────────────────────────────────────
    W = 12

    print("\n" + "=" * 76)
    print(f"Note: KITTI official IoU — Car@0.7 / Ped@0.5 / Cyc@0.5")
    print(f"{'Class':<12} {'AP@0.5':>{W}} {'AP@KIT-IoU':>{W}} {'AP@0.5:0.95':>{W}} {'Easy':>{W}} {'Mod':>{W}} {'Hard':>{W}}")
    print("-" * 76)
    for i, name in enumerate(CLASS_NAMES):
        easy = kitti_ap["Easy"][i]
        mod  = kitti_ap["Moderate"][i]
        hard = kitti_ap["Hard"][i]
        iou_str = f"@{KITTI_IOU_THR.get(i, 0.5):.1f}"
        print(f"{name:<12} {ap50[i]:>{W}.4f} {ap50_kit[i]:>{W}.4f} "
              f"{ap5095[i]:>{W}.4f} "
              f"{easy:>{W}.4f} {mod:>{W}.4f} {hard:>{W}.4f}")

    def _mean(lst):
        v = [a for a in lst if not math.isnan(a)]
        return float(np.mean(v)) if v else float("nan")

    print("-" * 76)
    print(f"{'mAP':<12} {_mean(ap50):>{W}.4f} {_mean(ap50_kit):>{W}.4f} "
          f"{_mean(ap5095):>{W}.4f} "
          f"{_mean(kitti_ap['Easy']):>{W}.4f} "
          f"{_mean(kitti_ap['Moderate']):>{W}.4f} "
          f"{_mean(kitti_ap['Hard']):>{W}.4f}")
    print("=" * 76)

    print("\n[Recall @ Distance Bins]  (IoU≥0.5, conf≥0.001)")
    header = f"{'Class':<12}" + "".join(f"{lb:>12}" for lb in DIST_LABELS)
    print(header)
    print("-" * (12 + 12 * len(DIST_BINS)))
    for name, bins in recall_d.items():
        row = f"{name:<12}"
        for b in bins:
            r = b["recall"]
            row += f"  {r*100:6.1f}% ({b['tp']}/{b['gt']})"
        print(row)
    print()

    # ── save ─────────────────────────────────────────────────────────────────
    out = {
        "AP50":          dict(zip(CLASS_NAMES, ap50)),
        "AP50_KITTI_IoU": dict(zip(CLASS_NAMES, ap50_kit)),
        "AP5095":        dict(zip(CLASS_NAMES, ap5095)),
        "mAP50":         _mean(ap50),
        "mAP50_KITTI_IoU": _mean(ap50_kit),
        "mAP5095":       _mean(ap5095),
        "KITTI_IoU_per_class": KITTI_IOU_THR,
        "KITTI":         {d: dict(zip(CLASS_NAMES, v)) for d, v in kitti_ap.items()},
        "KITTI_mAP":     {d: _mean(vals) for d, vals in kitti_ap.items()},
        "recall_by_distance": recall_d,
    }

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved → {args.save_json}")

    return out


if __name__ == "__main__":
    main()
