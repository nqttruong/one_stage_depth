"""Stage-2 preprocessing: compute median LiDAR Z per kept object.

For each (image_id, object) pair in the dataset we:
  1. Load Velodyne points (Nx4: x, y, z, reflectance).
  2. Transform to camera coord using Tr_velo_to_cam from the calib file.
  3. Keep points whose (x, y, z) lies inside the GT 3D bounding box.
  4. Take the median z. This is the per-object depth used at training time.

Result is written as a JSON whose keys are image ids and whose values are
lists of floats aligned with the order of objects KEPT by
`KITTIOGCDEDataset._build_targets` (same filtering: class, truncation,
occlusion, min bbox size). Use this with `--depth-source` in train.py.

Usage:
    python prepare_lidar_depth.py \
        --kitti-root /data/kitti/training \
        --split splits/kitti_train.txt \
        --out kitti_train_lidar_depth.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ogcde.dataset import KITTIOGCDEDataset, parse_kitti_calib, parse_kitti_label


# ----------------------------- KITTI calib helpers --------------------------

def parse_calib_full(path):
    """Like parse_kitti_calib but returns Tr_velo_to_cam and R0_rect too."""
    out = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = np.array([float(x) for x in v.split()], dtype=np.float64)
    P2 = out["P2"].reshape(3, 4)
    Tr = out["Tr_velo_to_cam"].reshape(3, 4)
    R0 = out["R0_rect"].reshape(3, 3)
    return P2, Tr, R0


def velo_to_cam(pts_velo, Tr, R0):
    """(N, 3 or 4) velodyne -> rectified camera coords."""
    pts = pts_velo[:, :3]
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_cam = pts_h @ Tr.T          # (N, 3)
    pts_rect = pts_cam @ R0.T       # apply rectification
    return pts_rect


# ----------------------------- bbox-3D test ---------------------------------

def points_in_3d_box(pts_cam, obj):
    """Return mask of points lying inside the GT 3D bbox.

    KITTI 3D box convention (rectified camera frame):
        center bottom = (x, y, z) = obj['loc']
        size (h, w, l) = obj['dims']     # height (Y), width (X), length (Z)
        ry around Y axis
    The box extends from y in [y_loc - h, y_loc] (Y is downward in cam coord).
    """
    x0, y0, z0 = obj["loc"]
    h, w, l = obj["dims"]
    ry = obj["rot_y"]

    # translate to box-local frame
    p = pts_cam - np.array([x0, y0, z0])

    # un-rotate around Y
    c, s = np.cos(-ry), np.sin(-ry)
    Rmat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    p = p @ Rmat.T

    # box bounds (Y is downward; KITTI loc is at the bottom face)
    in_x = (p[:, 0] >= -w / 2) & (p[:, 0] <= w / 2)
    in_y = (p[:, 1] >= -h)     & (p[:, 1] <= 0)
    in_z = (p[:, 2] >= -l / 2) & (p[:, 2] <= l / 2)
    return in_x & in_y & in_z


# ----------------------------- main loop ------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--velodyne-dir", default=None,
                    help="default: <kitti_root>/velodyne")
    ap.add_argument("--min-points", type=int, default=10,
                    help="If fewer points fall inside the 3D box, "
                         "fall back to obj['loc'][2] (bottom-z).")
    args = ap.parse_args()

    velo_dir = args.velodyne_dir or os.path.join(args.kitti_root, "velodyne")
    label_dir = os.path.join(args.kitti_root, "label_2")
    calib_dir = os.path.join(args.kitti_root, "calib")

    with open(args.split) as f:
        ids = [l.strip() for l in f if l.strip()]

    out = {}
    n_imgs = len(ids)
    n_kept_total = 0
    n_lidar_total = 0

    for i, fid in enumerate(ids):
        velo_path = os.path.join(velo_dir, f"{fid}.bin")
        if not os.path.exists(velo_path):
            print(f"[skip] no velodyne for {fid}")
            continue

        P2, Tr, R0 = parse_calib_full(os.path.join(calib_dir, f"{fid}.txt"))
        objects = parse_kitti_label(os.path.join(label_dir, f"{fid}.txt"))

        # Replicate the dataset's keep-filter so output order matches.
        kept = []
        for obj in objects:
            if obj["type"] not in KITTIOGCDEDataset.CLASS_MAP:
                continue
            if obj["truncated"] > 0.5 or obj["occluded"] > 2:
                continue
            l_, t_, r_, b_ = obj["bbox"]
            if (r_ - l_) < 5 or (b_ - t_) < 5:
                continue
            kept.append(obj)

        if not kept:
            out[fid] = []
            continue

        # Velodyne points
        pts = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
        # KITTI convention: drop points behind the lidar (x_velo > 0 is forward)
        pts = pts[pts[:, 0] > 0]
        pts_cam = velo_to_cam(pts, Tr, R0)

        per_object_depth = []
        for obj in kept:
            mask = points_in_3d_box(pts_cam, obj)
            if mask.sum() >= args.min_points:
                d = float(np.median(pts_cam[mask, 2]))
                n_lidar_total += 1
            else:
                d = float(obj["loc"][2])  # fallback
            per_object_depth.append(d)
            n_kept_total += 1

        out[fid] = per_object_depth

        if (i + 1) % 100 == 0 or (i + 1) == n_imgs:
            print(f"[{i+1}/{n_imgs}] {n_kept_total} objects, "
                  f"{n_lidar_total} ({100*n_lidar_total/max(1, n_kept_total):.1f}%) "
                  f"got LiDAR-median depth")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
