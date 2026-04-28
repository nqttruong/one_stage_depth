# OGCDE One-Stage

Reference implementation of a YOLO-style one-stage detector with geometry
head for monocular distance estimation.

## Core design (locked-in)

```
d  =  Z-depth (metric, from KITTI label_2 directly)
s  =  geometric correction ≈ sec(θ),   θ = angle from optical axis
distance  =  s · d  =  sqrt(x² + y² + z²)
```

Each object prediction:

```
(x, y, w, h)      bbox
obj, cls          detection scores
d                 depth     (log-activated: d = exp(d_raw))
s                 scale     (log-activated: s = exp(s_raw))
(dx, dy)          contact offset from bbox bottom center
distance = s · d  metric distance (geometry branch)
```

Pipeline = **YOLOv8n-scale backbone + PAN-FPN neck + extended anchor-free
head (9 + nc channels)** trained from scratch with multi-task loss + λ_geo
warmup. Center-radius assignment with ATSS-style size filter.

## Layout

```
ogcde_onestage/
  ogcde/
    model.py          backbone + neck + OGCDEHead + decode helpers
    loss.py           OGCDELoss + center-radius assigner (mutable lambdas)
    dataset.py        KITTI loader, color jitter, opt-in H-flip, depth_source
    metrics.py        AbsRel/RMSE/δ/DE/CPE/Stability + OGCDEEvaluator
    utils.py          letterbox, NMS, decode_predictions
  train.py            training entry point with λ_geo warmup
  evaluate_kitti.py   runs full OGCDE metric suite on a KITTI split
  inference.py        video/image-folder inference, includes Stability
  split_kitti.py      build train/val split files
  prepare_lidar_depth.py   stage-2 median-LiDAR-Z preprocessing
```

## Quick start

### Stage 1 — train fast with bottom-Z depth GT

```bash
python split_kitti.py --kitti-root /data/kitti/training --out-dir splits/

python train.py \
    --kitti-root /media/truong/01DBB45ECE0C4E00/dl/kitti_object/training \
    --train-split /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/kitti_train.txt \
    --val-split   /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/kitti_val.txt \
    --img-size 640 \
    --batch 16 \
    --epochs 100 \
    --workers 4 \
    --geo-warmup-epochs 10 \
    --save-dir runs/ogcde_v1
```

### Stage 2 — refine with median-LiDAR-Z depth GT

```bash
# Pre-compute per-object median LiDAR Z (one-time)
python prepare_lidar_depth.py \
    --kitti-root /data/kitti/training \
    --split splits/kitti_train.txt \
    --out cache/kitti_train_lidar.json

python prepare_lidar_depth.py \
    --kitti-root /data/kitti/training \
    --split splits/kitti_val.txt \
    --out cache/kitti_val_lidar.json

# Re-train with the better depth
python train.py \
    --kitti-root /data/kitti/training \
    --train-split splits/kitti_train.txt --val-split splits/kitti_val.txt \
    --depth-source cache/kitti_train_lidar.json \
    --epochs 50 --geo-warmup-epochs 5
# (use a separate file for val if you also want LiDAR-eval depth metrics)
```

### Evaluate

```bash
python evaluate_kitti.py \
    --ckpt runs/ogcde/best.pt \
    --kitti-root /data/kitti/training \
    --split splits/kitti_val.txt
```

### Run on video (with stability metric)

```bash
python inference.py --ckpt runs/ogcde/best.pt --source clip.mp4 --save-video
```

## Configuration decisions (locked-in)

| Item | Value | Rationale |
|---|---|---|
| `d` interpretation | metric Z, `d_gt = z` | No external dependency; clean GT from KITTI |
| `distance` | `sqrt(x²+y²+z²)` | Forces `s` to learn `sec(θ)` — geometry-aware insight |
| `d`/`s` activation | `exp` (log-space learning) | Strict positivity; stable gradients across orders of magnitude |
| `L_depth`, `L_scale` | log-space L1 | Same as activation; exponential-family-friendly |
| `L_geo` | linear L1 on meters | Direct distance error — what the paper claims to optimize |
| `λ_geo` schedule | linear 0.1 → 2.0 over 10 epochs | Avoids dominating early training |
| Stage-1 `d_gt` | `z` of bottom center | Fast, debuggable |
| Stage-2 `d_gt` | median LiDAR Z inside 3D box | More accurate for paper-grade numbers |
| Augmentation | Color jitter (default ON) | P2-safe |
| H-flip | Opt-in via `--hflip` | Geometry-safe in our 2D-target setup |
| Mosaic | Excluded | Breaks projection geometry |
| Backbone | From scratch, YOLOv8n-scale | Phase 1 stability; can port later |
| Assigner | Center-radius + ATSS size filter | Distance is the metric of interest, not box AP |
| Classes | 3 (Car incl. Van/Truck, Pedestrian incl. Person_sitting, Cyclist) | KITTI standard |

## Why H-flip is geometry-safe here (and why P2 doesn't need updating)

You might have read that flipping needs `P2[0,2] := W − P2[0,2]`. That's
true only if P2 is consumed *after* augmentation (e.g., re-projecting 3D
points or doing differentiable rendering). Our pipeline uses P2 exactly
once, at GT-load time, to project `loc` to 2D pixels. After that, the
target is `(boxes_2d, contact_2d, scalar dist, scalar depth)`. A
horizontal flip:

- mirrors `boxes_2d` and `contact_2d` correctly via `u → W − u`,
- doesn't touch `dist = ||xyz||` (invariant to x sign flip),
- doesn't touch `depth = z` (z is independent of x).

So all four targets remain consistent without ever re-touching P2. H-flip
is opt-in (`--hflip`) only because we wanted to be explicit about it.

## λ_geo warmup (implementation)

```python
def lambda_geo_schedule(epoch, warmup=10, start=0.1, end=2.0):
    if epoch >= warmup:
        return end
    return start + (end - start) * (epoch / warmup)
```

Called once per epoch in `train.py`; pushes the new value into the loss
via `criterion.set_lambda_geo(lg)`.

## Honest limitations / what's NOT in this code

1. **Stability metric requires sequence data.** KITTI training/val is
   single-frame. Use KITTI Tracking or your own video with the simple IoU
   tracker in `inference.py`. For real publishable numbers, swap in
   ByteTrack/OC-SORT.
2. **Backbone from scratch.** Phase-2 plan is to port to Ultralytics
   YOLOv8 with COCO weights — recipe sketched below, not wired in.
3. **TaskAlignedAssigner not implemented.** Center-radius is fine for
   distance learning; if you want to claim YOLOv8-parity AP, add it as an
   ablation row (~200 LOC inside `loss.py::assign_targets`).
4. **No principled objective re-weighting between depth/scale/geo.** Currently
   λ_geo is the only thing that's scheduled. Auto-balancing schemes
   (uncertainty weighting, GradNorm) are an open avenue.
5. **`prepare_lidar_depth.py` requires Velodyne `.bin` files.** Not all
   KITTI subsets ship them; if you only have label_2 + image_2 + calib,
   you can only do stage-1.

## Phase-2: Port to Ultralytics YOLOv8 (sketch, not wired in)

```python
from ultralytics.nn.tasks import DetectionModel
from ogcde.model import OGCDEHead

yolo = DetectionModel(cfg='yolov8n.yaml')
yolo.load('yolov8n.pt')
yolo.model[-1] = OGCDEHead(nc=3, ch=(64, 128, 256))   # replace Detect
```

Then `train.py`'s forward becomes `yolo.forward(x)`, and `OGCDELoss` works
unchanged.

## Citation key idea

> We reformulate monocular distance estimation as a one-stage detection
> problem, where geometry (contact point + scale) is learned implicitly
> via multi-task loss. Under `distance = s · d`, the depth branch learns
> metric Z while the scale branch absorbs the per-object geometric
> correction `sec(θ)` — yielding a single real-time pipeline that
> bypasses the depth-map → calibration cascade.

## What this is, and what it isn't

> This is **geometry-aware distance learning** — not depth estimation, not
> 3D detection. The model never produces a dense depth map and never
> regresses a 3D bounding box. It produces, per detected object, a single
> scalar metric distance with explicit per-object geometric structure.
