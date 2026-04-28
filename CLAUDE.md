# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

OGCDE ("One-stage Geometry-aware Contour-based Distance Estimation") is a YOLO-style one-stage detector that jointly predicts bounding boxes and metric distance for each object using monocular images. It is **not** a depth-map estimator and does **not** regress 3D bounding boxes. The key formula:

```
distance = s · d     where  d = Z-depth,  s ≈ sec(θ)
```

Both `d` and `s` are log-space predictions (`exp(d_raw)`, `exp(s_raw)`); `distance = exp(d_raw + s_raw)`.

## Commands

**Prepare data splits (one-time):**
```bash
python split_kitti.py --kitti-root /data/kitti/training --out-dir splits/
```

**Stage-1 training (bottom-Z depth GT, fast):**
```bash
python train.py \
    --kitti-root /data/kitti/training \
    --train-split splits/kitti_train.txt \
    --val-split splits/kitti_val.txt \
    --epochs 100 --batch 16 --geo-warmup-epochs 10
```

**Stage-2 training (median LiDAR-Z depth GT, better accuracy):**
```bash
# Pre-compute per-object depth (requires Velodyne .bin files)
python prepare_lidar_depth.py --kitti-root /data/kitti/training \
    --split splits/kitti_train.txt --out cache/train_lidar.json

python train.py --kitti-root /data/kitti/training \
    --train-split splits/kitti_train.txt --val-split splits/kitti_val.txt \
    --depth-source cache/train_lidar.json --epochs 50 --geo-warmup-epochs 5
```

**Evaluate:**
```bash
python evaluate_kitti.py --ckpt runs/ogcde/best.pt \
    --kitti-root /data/kitti/training --split splits/kitti_val.txt
```

**Inference on video or image folder:**
```bash
python inference.py --ckpt runs/ogcde/best.pt --source clip.mp4 --save-video
```

## Architecture

```
Input → Backbone (YOLOv8n-scale) → Neck (PAN-FPN) → OGCDEHead × 3 scales
                                                           ↓
                                               OGCDELoss (multi-task)
```

**`ogcde/model.py`** — All model code.
- `Backbone`: Conv/C2f/SPPF stack, outputs feature maps at strides 8/16/32. Width tuple `(16,32,64,128,256)` matches YOLOv8n scale.
- `Neck`: Standard PAN-FPN — top-down upsample + bottom-up downsample.
- `OGCDEHead`: Single head per scale, outputs `9 + nc` channels per spatial location: `[cx,cy,w,h | obj | cls×nc | d_raw | s_raw | dx,dy]`.
- Decode helpers (`split_pred`, `decode_bbox`, `decode_contact`, `decode_depth_scale`) are pure functions used by both training and inference.

**`ogcde/loss.py`** — `OGCDELoss` + `assign_targets`.
- Assignment is **center-radius + ATSS-style size filter**: a cell is positive if it's inside the GT box AND within `center_radius=1.5` stride-units of GT center, filtered by scale-appropriate stride. Falls back to nearest cell if no cell qualifies.
- Multi-task loss: `w_box·CIoU + w_obj·BCE + w_cls·BCE + λ1·L_depth + λ2·L_scale + λ3·L_contact + λ4·L_geo`. All geometry losses are log-space L1 except `L_geo` which is linear L1 on meters.
- `criterion.set_lambda_geo(value)` is called each epoch in `train.py` to implement the warmup schedule.

**`ogcde/dataset.py`** — `KITTIOGCDEDataset` + `collate_ogcde`.
- Expects KITTI directory structure: `image_2/`, `label_2/`, `calib/`. P2 is read from `calib/` and used **only at GT-load time** to project the 3D bottom center to 2D contact point.
- `collate_ogcde` concatenates per-image targets into flat tensors with a `batch_idx` vector (same pattern as Ultralytics YOLOv8).
- 3 classes: Car/Van/Truck → 0, Pedestrian/Person_sitting → 1, Cyclist → 2.
- Mosaic augmentation is intentionally excluded — it breaks projection geometry.

**`ogcde/metrics.py`** — `OGCDEEvaluator`. Computes AbsRel, RMSE, RMSE_log, δ1/δ2/δ3 (depth), DE and CPE (per-object, IoU>0.5 matched), and Stability (requires temporal sequence data).

## Key design constraints (do not change without understanding)

- **`d` is bottom-center Z** (not center Z, not euclidean distance). This comes directly from `loc[2]` in KITTI labels.
- **`distance = sqrt(x²+y²+z²)`** (euclidean). This forces `s` to learn `sec(θ)` — the geometric correction. Changing this breaks the geometric interpretation.
- **Exp activation** for `d` and `s` (log-space learning). The clamping in `decode_depth_scale` (`d_raw ∈ [-5,6]`, `s_raw ∈ [-3,3]`) is intentional for numerical stability.
- **H-flip is opt-in** (`--hflip`). It is geometry-safe (distance/depth/scale are invariant to x-flip; only 2D coords need mirroring). P2 is **not** updated on flip because it is only used at GT-load time, not after.
- **No mosaic**, no TaskAlignedAssigner — these are known omissions, not bugs.

## Checkpoint format

Saved as `{"epoch", "model", "optim", "val_loss", "args"}`. Load with:
```python
ckpt = torch.load("best.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
```

## Phase-2 Ultralytics port (not wired in)

The head can be dropped into a pretrained YOLOv8 by replacing its last module with `OGCDEHead`; `OGCDELoss` requires no changes. See README for the sketch.
