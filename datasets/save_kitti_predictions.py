"""Save OGCDE predictions in KITTI label format for official evaluation.

Output per image: runs/eval/predictions/XXXXXX.txt
Format (15 fields + score):
  type trunc occ alpha x1 y1 x2 y2 h w l x y z ry score

3D fields (h,w,l,x,y,z,ry) are set to dummy values since OGCDE
does not predict 3D boxes — only 2D detection is evaluated.

Usage:
    python save_kitti_predictions.py \\
        --ckpt runs/ogcde_lidar_p2/best.pt \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt \\
        --out-dir runs/eval/predictions \\
        --conf 0.001
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ogcde.model import OGCDENet
from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde
from ogcde.utils import decode_predictions, unletterbox_boxes
from ogcde.metrics import xywh_to_xyxy

CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",        required=True)
    ap.add_argument("--kitti-root",  required=True)
    ap.add_argument("--split",       required=True)
    ap.add_argument("--out-dir",     default="runs/eval/predictions")
    ap.add_argument("--img-size",    type=int,   default=640)
    ap.add_argument("--batch",       type=int,   default=16)
    ap.add_argument("--workers",     type=int,   default=4)
    ap.add_argument("--conf",        type=float, default=0.001)
    ap.add_argument("--nms-iou",     type=float, default=0.5)
    ap.add_argument("--num-classes", type=int,   default=3)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    bs    = ckpt.get("args", {}).get("backbone_size", "n")
    model = OGCDENet(nc=args.num_classes, backbone_size=bs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = KITTIOGCDEDataset(
        args.kitti_root, args.split,
        img_size=args.img_size, augment=False,
        filter_difficult=False,   # keep all for full recall
    )
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=collate_ogcde,
    )

    n_written = 0
    with torch.no_grad():
        for imgs, targets, meta in loader:
            imgs  = imgs.to(device, non_blocking=True)
            preds = model(imgs)
            dets  = decode_predictions(
                preds, args.num_classes,
                obj_thr=args.conf, iou_thr=args.nms_iou,
            )

            for b, det in enumerate(dets):
                m      = meta[b]
                img_id = m["image_id"]

                pred_boxes = unletterbox_boxes(
                    det["boxes"].cpu().numpy(), m["ratio"], m["pad"])
                pred_xyxy  = xywh_to_xyxy(pred_boxes)   # x1,y1,x2,y2
                scores     = det["scores"].cpu().numpy()
                classes    = det["classes"].cpu().numpy()

                lines = []
                for i in range(len(scores)):
                    cls_id = int(classes[i])
                    name   = CLASS_NAMES.get(cls_id, "DontCare")
                    x1, y1, x2, y2 = pred_xyxy[i]
                    score  = float(scores[i])
                    # KITTI format: type trunc occ alpha x1 y1 x2 y2 h w l x y z ry score
                    line = (f"{name} -1 -1 -10 "
                            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                            f"-1 -1 -1 -1000 -1000 -1000 -10 {score:.4f}")
                    lines.append(line)

                out_path = os.path.join(args.out_dir, f"{img_id}.txt")
                with open(out_path, "w") as f:
                    f.write("\n".join(lines))
                n_written += 1

    print(f"Saved {n_written} prediction files → {args.out_dir}/")


if __name__ == "__main__":
    main()
