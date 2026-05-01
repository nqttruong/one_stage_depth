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
    ap.add_argument("--backbone-size", default="n", choices=["n", "m"],
                    help="YOLOv8 backbone scale: n (default) or m.")
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
    # geometry loss lambdas
    ap.add_argument("--w-cp", type=float, default=1.0,
                    help="λ_contact weight (default 1.0). Set lower (e.g. 0.3) to "
                         "reduce contact-point dominance in total loss.")
    ap.add_argument("--focal-gamma", type=float, default=1.5,
                    help="Focal loss gamma for obj/cls BCE (0 = plain BCE).")
    ap.add_argument("--cls-weights", type=float, nargs="+", default=None,
                    help="Per-class loss weights e.g. '1.0 3.0 5.0' for Car/Ped/Cyclist.")
    # resume
    ap.add_argument("--resume", default=None,
                    help="Path to checkpoint to resume from (.pt).")
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
    model = OGCDENet(nc=args.num_classes, backbone_size=args.backbone_size).to(device)
    criterion = OGCDELoss(
        nc=args.num_classes,
        lambdas=(1.0, 0.5, args.w_cp, 2.0),
        det_weights=(args.w_box, args.w_obj, args.w_cls),
        has_depth_gt=not args.no_depth_gt,
        focal_gamma=args.focal_gamma,
        cls_weights=args.cls_weights,
    ).to(device)

    # Load pretrained backbone before optimizer init (so frozen params are excluded)
    if args.pretrained_backbone and not args.resume:
        model.load_pretrained_backbone(args.pretrained_backbone)
        if args.freeze_backbone_epochs > 0:
            for p in model.backbone.parameters():
                p.requires_grad_(False)
            print(f"[pretrained] backbone frozen for first {args.freeze_backbone_epochs} epochs")

    # Neck+Head params always train; backbone params added back after freeze period
    neck_head_params = list(model.neck.parameters()) + list(model.head.parameters())
    backbone_params  = list(model.backbone.parameters())
    if args.pretrained_backbone and not args.resume and args.freeze_backbone_epochs > 0:
        optimizer = torch.optim.AdamW(
            neck_head_params, lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt_r = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_r["model"])
        start_epoch = ckpt_r["epoch"] + 1
        best_val = ckpt_r.get("best_val", ckpt_r["val_loss"])
        # Scheduler restarted fresh (last_epoch=-1 = LR starts at args.lr).
        # Do NOT fast-forward: the old run may have used a different T_max,
        # fast-forwarding would reset LR to near-max and cause divergence.
        # Không load optimizer state: khi loss weights thay đổi, Adam's second
        # moment (v) từ run cũ sẽ gây NaN do step size bất thường. Fresh optimizer
        # với LR mới là an toàn hơn cho fine-tuning.
        print(f"Resumed weights from {args.resume} "
              f"(epoch {ckpt_r['epoch']} → continue from {start_epoch}, "
              f"val_loss={ckpt_r['val_loss']:.3f}, best_val={best_val:.3f})"
              f" — optimizer reset, lr={args.lr}")

    train_loader, val_loader = build_loaders(args)

    for epoch in range(start_epoch, args.epochs):
        # ------------------- unfreeze backbone ---------------
        if (args.pretrained_backbone and not args.resume
                and args.freeze_backbone_epochs > 0
                and epoch == args.freeze_backbone_epochs):
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            optimizer.add_param_group({
                "params": backbone_params,
                "lr": args.lr * 0.1,
                "weight_decay": args.weight_decay,
            })
            print(f"[epoch {epoch+1:03d}] backbone unfrozen (lr={args.lr*0.1:.2e})")

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

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ({loss.item():.3g}), skipping batch")
                scaler.update()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            if not torch.isfinite(grad_norm):
                print(f"  [warn] non-finite grad norm, skipping update")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()

            for k in run:
                run[k] += comp[k]
            n_iters += 1

        scheduler.step()
        dt = time.time() - t0
        if n_iters == 0:
            print(f"[epoch {epoch+1:03d}] λ_geo={lg:.2f} train {dt:.1f}s | [warn] all batches skipped")
        else:
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
        is_best = vloss < best_val
        if is_best:
            best_val = vloss
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "val_loss": vloss,
            "best_val": best_val,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print(f"  -> new best ({best_val:.3f})")


if __name__ == "__main__":
    main()
