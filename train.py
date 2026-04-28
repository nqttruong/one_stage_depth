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

import torch
from torch.utils.data import DataLoader

from ogcde.model import OGCDENet
from ogcde.loss import OGCDELoss
from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde


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
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--train-split", required=True)
    ap.add_argument("--val-split", required=True)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="runs/ogcde")
    ap.add_argument("--distance-mode", default="euclidean",
                    choices=["euclidean", "depth"])
    ap.add_argument("--depth-mode", default="bottom_z",
                    choices=["bottom_z", "center_z"])
    ap.add_argument("--depth-source", default=None,
                    help="Optional path to per-object depth JSON/npy "
                         "(stage-2 LiDAR-median depth).")
    ap.add_argument("--no-depth-gt", action="store_true",
                    help="Train without L_depth / L_scale (pure L_geo).")
    ap.add_argument("--hflip", action="store_true",
                    help="Enable horizontal flip aug (geometry-safe in our setup).")
    # detection loss weights
    ap.add_argument("--w-box", type=float, default=7.5)
    ap.add_argument("--w-obj", type=float, default=1.0)
    ap.add_argument("--w-cls", type=float, default=0.5)
    ap.add_argument("--focal-gamma", type=float, default=1.5,
                    help="Focal loss gamma for obj/cls BCE (0 = plain BCE).")
    # resume
    ap.add_argument("--resume", default=None,
                    help="Path to checkpoint to resume from (.pt).")
    # λ_geo warmup
    ap.add_argument("--geo-warmup-epochs", type=int, default=10)
    ap.add_argument("--geo-start", type=float, default=0.1)
    ap.add_argument("--geo-end", type=float, default=2.0)
    return ap.parse_args()


def build_loaders(args):
    train_ds = KITTIOGCDEDataset(
        args.kitti_root, args.train_split,
        img_size=args.img_size, augment=True,
        distance_mode=args.distance_mode, depth_mode=args.depth_mode,
        hflip=args.hflip, depth_source=args.depth_source,
    )
    val_ds = KITTIOGCDEDataset(
        args.kitti_root, args.val_split,
        img_size=args.img_size, augment=False,
        distance_mode=args.distance_mode, depth_mode=args.depth_mode,
        depth_source=args.depth_source,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate_ogcde, drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=collate_ogcde,
        pin_memory=True,
    )
    return train_loader, val_loader


def main():
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = OGCDENet(nc=args.num_classes).to(device)
    criterion = OGCDELoss(
        nc=args.num_classes,
        det_weights=(args.w_box, args.w_obj, args.w_cls),
        has_depth_gt=not args.no_depth_gt,
        focal_gamma=args.focal_gamma,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    if args.resume:
        ckpt_r = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_r["model"])
        # Không load optimizer state: khi loss weights thay đổi, Adam's second
        # moment (v) từ run cũ sẽ gây NaN do step size bất thường. Fresh optimizer
        # với LR mới là an toàn hơn cho fine-tuning.
        print(f"Resumed weights from {args.resume} "
              f"(epoch {ckpt_r['epoch']}, val_loss={ckpt_r['val_loss']:.3f})"
              f" — optimizer reset, lr={args.lr}")

    train_loader, val_loader = build_loaders(args)

    best_val = float("inf")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # ------------------- λ_geo warmup -----------------
        lg = lambda_geo_schedule(
            epoch,
            warmup_epochs=args.geo_warmup_epochs,
            start=args.geo_start,
            end=args.geo_end,
        )
        criterion.set_lambda_geo(lg)

        # ------------------- train ------------------------
        model.train()
        t0 = time.time()
        run = {"total": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0,
               "depth": 0.0, "scale": 0.0, "contact": 0.0, "geo": 0.0}
        n_iters = 0

        for imgs, targets, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = {k: (v.to(device) if torch.is_tensor(v) else v)
                       for k, v in targets.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                preds = model(imgs)
                loss, comp = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            for k in run:
                run[k] += comp[k]
            n_iters += 1

        scheduler.step()
        dt = time.time() - t0
        print(
            f"[epoch {epoch+1:03d}] λ_geo={lg:.2f} train {dt:.1f}s"
            f" | total {run['total']/n_iters:.3f}"
            f" box {run['box']/n_iters:.3f}"
            f" obj {run['obj']/n_iters:.3f}"
            f" cls {run['cls']/n_iters:.3f}"
            f" d {run['depth']/n_iters:.3f}"
            f" s {run['scale']/n_iters:.3f}"
            f" cp {run['contact']/n_iters:.3f}"
            f" geo {run['geo']/n_iters:.3f}"
        )

        # ------------------- validate ---------------------
        model.eval()
        vloss = 0.0
        vn = 0
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = {k: (v.to(device) if torch.is_tensor(v) else v)
                           for k, v in targets.items()}
                preds = model(imgs)
                loss, _ = criterion(preds, targets)
                vloss += loss.item()
                vn += 1
        vloss = vloss / max(1, vn)
        print(f"[epoch {epoch+1:03d}] val total={vloss:.3f}")

        # ------------------- checkpoint -------------------
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "val_loss": vloss,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))
        if vloss < best_val:
            best_val = vloss
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print(f"  -> new best ({best_val:.3f})")


if __name__ == "__main__":
    main()
