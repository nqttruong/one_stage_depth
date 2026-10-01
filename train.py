"""Training script for OGCDE one-stage.

Example:
    python train.py \
        --kitti-root /data/kitti/training \
        --train-split splits/kitti_train.txt \
        --val-split   splits/kitti_val.txt \
        --img-size 640 --batch 16 --epochs 100

Tip: KITTI Eigen split files can be generated with split_kitti.py (below).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ogcde.datasets import KITTIDistanceDataset, collate_distance_batch
from ogcde.detectors import build_detector
from ogcde.distance import build_distance_method
from ogcde.distance.base import DistanceDetectionModel


def lambda_geo_schedule(epoch: int, warmup_epochs: int = 10,
                        start: float = 0.1, end: float = 2.0) -> float:
    """Linear warmup of λ_geo from `start` to `end` over `warmup_epochs`.

    Rationale: early in training the bbox + contact branches are random,
    so distance_pred = exp(s_raw + d_raw) is essentially uniform. With high
    λ_geo this dominates the gradient and starves the detection branch.
    Warming λ_geo from 0.1 → 2.0 keeps detection learning first, then
    progressively tightens the geometry constraint.
    """
    if epoch >= warmup_epochs:
        return end
    return start + (end - start) * (epoch / max(1, warmup_epochs))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root",   default=None)
    ap.add_argument("--train-split",  default=None)
    ap.add_argument("--val-split",    default=None)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--backbone-size", default="n", choices=["n", "m"],
                    help="YOLOv8 backbone scale: n (default) or m.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="runs/ogcde")
    ap.add_argument("--distance-mode", default="euclidean",
                    choices=["euclidean", "depth"])
    ap.add_argument("--detector", default="yolov8_custom",
                    choices=["yolov8_custom", "fcos"],
                    help="New detector interface selector. Legacy training path stays active for the current YOLO implementation.")
    ap.add_argument("--distance-method", default="learned_scale",
                    choices=["direct_distance", "learned_scale", "analytic_geometry", "geometry_residual"],
                    help="Distance module for the detector-agnostic refactor.")
    ap.add_argument("--depth-mode", default="bottom_z",
                    choices=["bottom_z", "center_z"])
    ap.add_argument("--depth-source", default=None,
                    help="Optional path to per-object depth JSON/npy "
                         "(stage-2 LiDAR-median depth).")
    ap.add_argument("--dist-source", default=None,
                    help="Optional path to lidar_gt JSON from prepare_lidar_gt.py "
                         "to override distance GT with LiDAR-based values.")
    ap.add_argument("--max-grad-norm", type=float, default=10.0,
                    help="Gradient clipping max norm (default 10.0, use 2.0 for noisy GT).")
    ap.add_argument("--no-depth-gt", action="store_true",
                    help="Train without L_depth / L_scale (pure L_geo).")
    ap.add_argument("--hflip", action="store_true",
                    help="Enable horizontal flip aug (geometry-safe in our setup).")
    ap.add_argument("--strong-aug", action="store_true",
                    help="Stronger color jitter (0.4) + random crop (P=0.5).")
    # detection loss weights
    ap.add_argument("--w-box", type=float, default=7.5)
    ap.add_argument("--w-obj", type=float, default=1.0)
    ap.add_argument("--w-cls", type=float, default=0.5)
    # geometry loss lambdas
    ap.add_argument("--w-cp", type=float, default=1.0,
                    help="λ_contact weight (default 1.0).")
    ap.add_argument("--focal-gamma", type=float, default=1.5,
                    help="Focal loss gamma for obj/cls BCE (0 = plain BCE).")
    ap.add_argument("--cls-weights", type=float, nargs="+", default=None,
                    help="Per-class loss weights e.g. '1.0 3.0 5.0' for Car/Ped/Cyclist.")
    # ablation flags
    ap.add_argument("--no-s-head", action="store_true",
                    help="Ablation V1/V2: disable sec(θ) head — distance ≡ depth.")
    ap.add_argument("--no-contact", action="store_true",
                    help="Ablation V1: disable contact point supervision.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility (set same across ablation runs).")
    ap.add_argument("--config", default=None,
                    help="Optional YAML config file; values override argparse defaults.")
    # resume
    ap.add_argument("--resume", default=None,
                    help="Path to checkpoint to resume from (.pt).")
    ap.add_argument("--reset-best", action="store_true",
                    help="Reset best_val to inf when resuming (useful when val_loss "
                         "was best at early epoch due to bad training dynamics).")
    # pretrained backbone
    ap.add_argument("--pretrained-backbone", default=None, metavar="PATH",
                    help="Path to yolov8n.pt. Transplants backbone weights before "
                         "training. Ignored when --resume is set.")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=10,
                    help="Freeze backbone for this many epochs so Neck+Head "
                         "stabilise first. 0 = no freezing.")
    # λ_geo warmup
    ap.add_argument("--geo-warmup-epochs", type=int, default=10)
    ap.add_argument("--geo-start", type=float, default=0.1)
    ap.add_argument("--geo-end", type=float, default=2.0)
    return ap.parse_args()


def build_loaders(args):
    train_ds = KITTIDistanceDataset(
        args.kitti_root,
        args.train_split,
        img_size=args.img_size,
        augment=True,
        hflip=args.hflip,
        filter_difficult=True,
        gt_source="annotation_bottom_center",
    )
    val_ds = KITTIDistanceDataset(
        args.kitti_root,
        args.val_split,
        img_size=args.img_size,
        augment=False,
        hflip=False,
        filter_difficult=True,
        gt_source="annotation_bottom_center",
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate_distance_batch, drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=collate_distance_batch,
        pin_memory=True,
    )
    return train_loader, val_loader


def apply_structured_config(args, cfg):
    if not isinstance(cfg, dict):
        return args
    for section_name in ("dataset", "detector", "distance", "training", "augmentation", "experiment"):
        section = cfg.get(section_name)
        if isinstance(section, dict):
            for key, value in section.items():
                if section_name == "detector" and key == "name":
                    setattr(args, "detector", value)
                elif section_name == "distance" and key == "method":
                    setattr(args, "distance_method", value)
                elif section_name == "dataset" and key == "gt_source":
                    setattr(args, "gt_source", value)
                else:
                    setattr(args, key.replace("-", "_"), value)
    for key, value in cfg.items():
        if key in {"dataset", "detector", "distance", "training", "augmentation", "experiment"}:
            continue
        if isinstance(value, dict):
            continue
        if hasattr(args, key.replace("-", "_")):
            setattr(args, key.replace("-", "_"), value)
    return args


def main():
    args = parse_args()

    if args.config is not None:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        args = apply_structured_config(args, cfg)

    for required in ("kitti_root", "train_split", "val_split"):
        if getattr(args, required) is None:
            raise ValueError(f"--{required.replace('_','-')} is required (set via CLI or --config YAML)")

    device = torch.device(args.device)
    detector = build_detector(args.detector, nc=args.num_classes, backbone_size=args.backbone_size)
    distance_method = build_distance_method(args.distance_method)
    model = DistanceDetectionModel(detector=detector, distance_method=distance_method).to(device)
    print(f"[config] detector={args.detector} distance={args.distance_method}")
    print(f"[config] model={type(model.detector).__name__} distance_method={type(model.distance_method).__name__}")

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader, val_loader = build_loaders(args)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        for batch in train_loader:
            imgs = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"]
            metas = batch["metas"]
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = model.compute_loss(logits, targets, metas)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered in epoch {epoch}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            running_loss += float(loss.detach().item())
            running_count += 1
        scheduler.step()
        avg_train = running_loss / max(1, running_count)
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["images"].to(device, non_blocking=True)
                targets = batch["targets"]
                metas = batch["metas"]
                logits = model(imgs)
                loss = model.compute_loss(logits, targets, metas)
                val_loss += float(loss.item())
                val_count += 1
        val_loss = val_loss / max(1, val_count)
        print(f"[epoch {epoch + 1:03d}] train_loss={avg_train:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            out = os.path.join(args.save_dir, "best.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args)}, out)

if __name__ == "__main__":
    main()
