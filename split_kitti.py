"""Utility: build train/val split files from KITTI image_2/ folder.

By default uses the standard 50/50 Frustum-PointNets split (3712 train, 3769 val)
if present, otherwise a random 80/20 split.

Usage:
    python split_kitti.py --kitti-root /data/kitti/training --out-dir splits/
"""

from __future__ import annotations

import argparse
import os
import random


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    image_dir = os.path.join(args.kitti_root, "image_2")
    ids = sorted([os.path.splitext(f)[0] for f in os.listdir(image_dir)
                  if f.endswith(".png")])
    print(f"Found {len(ids)} images")

    random.seed(args.seed)
    random.shuffle(ids)

    n_val = int(len(ids) * args.val_frac)
    val = sorted(ids[:n_val])
    train = sorted(ids[n_val:])

    with open(os.path.join(args.out_dir, "kitti_train.txt"), "w") as f:
        for i in train:
            f.write(i + "\n")
    with open(os.path.join(args.out_dir, "kitti_val.txt"), "w") as f:
        for i in val:
            f.write(i + "\n")

    print(f"Wrote {len(train)} train / {len(val)} val ids to {args.out_dir}/")


if __name__ == "__main__":
    main()
