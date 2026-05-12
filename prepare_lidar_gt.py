"""Extract LiDAR-based ground-truth distances following Zhu et al. / DistFormer.

For each annotated object:
  1. Load velodyne point cloud (.bin)
  2. Transform points to rectified camera coordinates (R0_rect @ Tr_velo_to_cam)
  3. Collect all points inside the object's 3D bounding box
  4. Sort by Euclidean distance from camera origin
  5. Pick the 10th-percentile point as GT distance
  6. Fall back to annotation distance if box has < min_points LiDAR hits

Output JSON format:
  {
    "000001": [
      {"obj_idx": 0, "type": "Car", "dist_lidar": 12.34, "dist_annot": 12.1, "n_points": 87},
      ...
    ],
    ...
  }

Usage:
    python prepare_lidar_gt.py \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt \\
        --out cache/lidar_gt_val.json

    python prepare_lidar_gt.py \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_train.txt \\
        --out cache/lidar_gt_train.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

# KITTI class groups (same as dataset.py)
_KEEP_CLASSES = {
    "Car": 0, "Van": 0, "Truck": 0,
    "Pedestrian": 1, "Person_sitting": 1,
    "Cyclist": 2,
}


# ─────────────────────────── calibration ────────────────────────────────────

def read_calib(calib_path: str) -> dict:
    data = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            data[key.strip()] = np.array([float(x) for x in val.split()])
    calib = {}
    calib["P2"]          = data["P2"].reshape(3, 4)
    calib["R0_rect"]     = data["R0_rect"].reshape(3, 3)
    calib["Tr_velo_cam"] = data["Tr_velo_to_cam"].reshape(3, 4)
    return calib


def velo_to_cam(points_v: np.ndarray, calib: dict) -> np.ndarray:
    """Transform (N,3) velodyne points to rectified camera coords (N,3)."""
    Tr = calib["Tr_velo_cam"]   # (3,4)
    R0 = calib["R0_rect"]       # (3,3)

    # homogeneous velodyne coords
    ones = np.ones((len(points_v), 1))
    pts_h = np.hstack([points_v, ones])       # (N,4)

    # velodyne → camera 0
    pts_c0 = (Tr @ pts_h.T).T                 # (N,3)

    # rectify
    pts_rect = (R0 @ pts_c0.T).T             # (N,3)
    return pts_rect


# ─────────────────────────── 3-D box membership ─────────────────────────────

def points_in_box(pts_cam: np.ndarray, h: float, w: float, l: float,
                  cx: float, cy: float, cz: float, ry: float) -> np.ndarray:
    """Return boolean mask of points inside a KITTI 3D bounding box.

    KITTI label convention:
      (cx, cy, cz) = bottom-center of box in rectified camera coords
      Box 3D center = (cx, cy - h/2, cz)
      ry = rotation around camera Y-axis (yaw)
      l/w/h = length (Z) / width (X) / height (Y) in object frame
    """
    # translate to box center
    dx = pts_cam[:, 0] - cx
    dy = pts_cam[:, 1] - (cy - h / 2)
    dz = pts_cam[:, 2] - cz

    # rotate around Y by -ry to align with object frame
    cos_r, sin_r = math.cos(-ry), math.sin(-ry)
    dx_r = dx * cos_r - dz * sin_r
    dz_r = dx * sin_r + dz * cos_r

    inside = (np.abs(dx_r) <= l / 2) & (np.abs(dy) <= h / 2) & (np.abs(dz_r) <= w / 2)
    return inside


# ─────────────────────────── per-image processing ───────────────────────────

def process_image(img_id: str, kitti_root: str, percentile: float = 0.10,
                  min_points: int = 1) -> list[dict]:
    calib_path = os.path.join(kitti_root, "calib", f"{img_id}.txt")
    label_path = os.path.join(kitti_root, "label_2", f"{img_id}.txt")
    velo_path  = os.path.join(kitti_root, "velodyne", f"{img_id}.bin")

    if not os.path.exists(velo_path):
        return []

    calib = read_calib(calib_path)

    # load velodyne (x,y,z,intensity), keep only x,y,z
    pts_v = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)[:, :3]

    # keep points in front of sensor (x > 0 in velodyne = forward)
    pts_v = pts_v[pts_v[:, 0] > 0]

    # transform to rectified camera coords
    pts_cam = velo_to_cam(pts_v, calib)

    # keep points with positive Z (in front of camera)
    pts_cam = pts_cam[pts_cam[:, 2] > 0]

    # precompute Euclidean distances from camera origin
    dists = np.sqrt((pts_cam ** 2).sum(axis=1))

    records = []
    with open(label_path) as f:
        for obj_idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            obj_type = parts[0]
            if obj_type == "DontCare":
                continue
            if obj_type not in _KEEP_CLASSES:
                continue

            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            cx, cy, cz = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])

            # annotation-based Euclidean distance
            dist_annot = math.sqrt(cx**2 + cy**2 + cz**2)

            # skip objects behind camera
            if cz <= 0:
                continue

            mask   = points_in_box(pts_cam, h, w, l, cx, cy, cz, ry)
            in_pts = pts_cam[mask]
            n_pts  = len(in_pts)

            if n_pts < min_points:
                # not enough LiDAR hits — use annotation as fallback
                records.append({
                    "obj_idx":    obj_idx,
                    "type":       obj_type,
                    "class_id":   _KEEP_CLASSES[obj_type],
                    "dist_lidar": dist_annot,   # fallback
                    "dist_annot": dist_annot,
                    "n_points":   0,
                    "fallback":   True,
                })
                continue

            in_dists = dists[mask]
            in_dists_sorted = np.sort(in_dists)
            idx = max(0, int(percentile * n_pts))
            dist_lidar = float(in_dists_sorted[idx])

            records.append({
                "obj_idx":    obj_idx,
                "type":       obj_type,
                "class_id":   _KEEP_CLASSES[obj_type],
                "dist_lidar": dist_lidar,
                "dist_annot": dist_annot,
                "n_points":   n_pts,
                "fallback":   False,
            })

    return records


# ─────────────────────────── main ───────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--split",      required=True)
    ap.add_argument("--out",        required=True)
    ap.add_argument("--percentile", type=float, default=0.10,
                    help="LiDAR percentile to pick (default 0.10 = 10th)")
    ap.add_argument("--min-points", type=int, default=1,
                    help="Min LiDAR hits required; fallback to annotation otherwise")
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.split) if l.strip()]
    print(f"Processing {len(ids)} images ...")

    out = {}
    fallback_total = 0
    obj_total = 0

    for i, img_id in enumerate(ids):
        records = process_image(img_id, args.kitti_root,
                                percentile=args.percentile,
                                min_points=args.min_points)
        out[img_id] = records
        fallback_total += sum(1 for r in records if r["fallback"])
        obj_total      += len(records)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(ids)} done ...")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)

    print(f"\nDone. {obj_total} objects total.")
    print(f"  LiDAR hits    : {obj_total - fallback_total} ({(obj_total-fallback_total)/max(obj_total,1)*100:.1f}%)")
    print(f"  Fallback (ann): {fallback_total} ({fallback_total/max(obj_total,1)*100:.1f}%)")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
