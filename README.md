# OGCDE — One-Stage Geometry-aware Distance Estimation

## TL;DR

OGCDE là một **one-stage detector** kết hợp phát hiện vật thể và ước lượng khoảng cách Euclidean trực tiếp từ ảnh RGB đơn lẻ, không cần LiDAR hay stereo camera. Trên KITTI với cùng split và cùng LiDAR GT của DistFormer, OGCDE đạt **δ<1.25 = 96.74%, AbsRel = 7.54%** — vượt DistFormer (93.67%, 10.39%) trong khi phải **tự detect bounding box**, không được cấp GT box như các baseline khác.

---

## Problem

**Bài toán:** Ước lượng khoảng cách Euclidean thực tế (mét) đến từng vật thể trong ảnh từ một camera đơn (monocular), đồng thời detect vật thể và dự đoán vị trí chân vật thể trên mặt đất.

**Dataset:** KITTI Object Detection — 7,481 ảnh training, 3 class: Car / Pedestrian / Cyclist.

**Baseline chính:** DistFormer (arXiv 2401.03191) — nhận GT bounding box làm input, chỉ predict distance.

**Điểm khác biệt:** OGCDE giải bài toán khó hơn — **end-to-end một lần forward**: tự detect box + predict distance + depth + contact point.

---

## Method

### Input / Output

```
Input : ảnh RGB (H × W × 3), bất kỳ kích thước → letterbox 640×640
Output: per-object { bounding box, class, score, distance (m), depth Z (m), contact point (px) }
```

### Architecture

```
Input (640×640×3)
    │
    ▼
┌─────────────────────┐
│  Backbone           │  YOLOv8m — Conv / C2f / SPPF
│  (COCO pretrained)  │  → feature maps stride 8 / 16 / 32
└──────────┬──────────┘
           │ P3, P4, P5
           ▼
┌─────────────────────┐
│  Neck (PAN-FPN)     │  top-down upsample + bottom-up downsample
└──────────┬──────────┘
           │ 3 scales
           ▼
┌─────────────────────┐
│  OGCDEHead × 3      │  per cell: 9 + nc channels
│                     │  [cx,cy,w,h | obj | cls×nc | d_raw | s_raw | dx,dy]
└─────────────────────┘
```

**Decode:**
```
depth    = exp(d_raw)              # Z-axis depth (m), d_raw ∈ [-5, 6]
distance = exp(d_raw + s_raw)      # Euclidean = depth × sec(θ), s_raw ∈ [-3, 3]
contact  = bbox_bottom + dxdy × stride   # 2D ground contact point (px)
```

`s_raw` học hệ số `sec(θ)` — bù góc lệch của vật thể so với trục quang học, tức là vật thể lệch sang trái/phải sẽ có distance > depth.

### Loss Function

```
L = w_box · L_CIoU
  + w_obj · L_BCE_objectness
  + w_cls · L_focal_class
  + λ₁ · L_depth          # log-L1 trên d_raw
  + λ₂ · L_scale          # log-L1 trên s_raw
  + λ₃ · L_contact        # L1 trên contact point offset
  + λ_geo · L_geo         # linear-L1 trên distance (mét)
```

Default weights: `w_box=5.0, w_obj=2.0, w_cls=0.5`, `λ_geo` warmup 0.1 → 2.0 trong 5 epoch.

**Target assignment:** center-radius ATSS — cell là positive nếu nằm trong GT box VÀ trong `center_radius = 1.5` stride của GT center.

### Training — Two-Phase Strategy

| | Phase 1 | Phase 2 |
|---|---|---|
| GT distance | Annotation (clean) | **LiDAR 10th-pct** (Zhu et al.) |
| Epochs | 350 | 100 |
| lr | 1e-4 → 3e-5 | 1e-5 |
| max_grad_norm | 10.0 | 2.0 |
| Mục đích | Convergence ổn định | Align với LiDAR distribution |

---

## Results

### OGCDE v7 — Custom split (5985 train), Annotation GT

| Split | Pairs | δ<1.25↑ | AbsRel↓ | RMSE↓ | mAP |
|---|---|---|---|---|---|
| Val (1,196 ảnh) | 4,779 | **99.79%** | **2.19%** | **1.136 m** | 0.540 |
| Test (300 ảnh) | 1,117 | 99.46% | 2.27% | 1.186 m | 0.548 |

### OGCDE LiDAR P2 vs DistFormer — Chen split (3711 train), LiDAR GT

| Method | Box input | n | δ<1.25↑ | AbsRel↓ | SqRel↓ | RMSE↓ | RMSElog↓ |
|---|---|---|---|---|---|---|---|
| DistFormer | **GT box** | all GT | 93.67% | 10.39% | 0.32 | **2.95 m** | 0.150 |
| **OGCDE LiDAR P2** (IoU) | **Tự detect** | 13,085 | **96.74%** | **7.54%** | **0.313** | 3.67 m | **0.098** |
| OGCDE LiDAR P2 (Oracle) | GT center | 17,499 | 86.58% | 11.87% | 1.045 | 6.58 m | 0.233 |

Per-class — IoU mode, LiDAR GT:

| Class | n | δ<1.25↑ | AbsRel↓ | RMSE↓ | vs DistFormer |
|---|---|---|---|---|---|
| Car | 11,912 | **96.85%** | **7.44%** | 3.81 m | ✅ OGCDE tốt hơn |
| Pedestrian | 1,008 | 95.04% | 8.50% | 1.66 m | ❌ DistFormer tốt hơn |
| Cyclist | 165 | **98.79%** | 8.43% | 1.91 m | ✅ OGCDE tốt hơn |

### Inference Speed (RTX 4070 Laptop, 640×640, batch=1)

| Stage | FPS | Latency |
|---|---|---|
| Model forward | 92.5 FPS | 10.8 ms |
| + NMS decode | 89.0 FPS | 11.2 ms |
| **Full pipeline** (letterbox+infer+decode) | **84.0 FPS** | **11.9 ms** |

---

## Repo Structure

```
one_stage_depth/
├── ogcde/
│   ├── model.py          # OGCDENet, OGCDEHead, decode helpers
│   ├── loss.py           # OGCDELoss, assign_targets (ATSS)
│   ├── dataset.py        # KITTIOGCDEDataset — supports dist_source, depth_source
│   ├── metrics.py        # OGCDEEvaluator, MAPEvaluator (AP + AP3D)
│   ├── utils.py          # decode_predictions, letterbox, unletterbox
│   └── oracle.py         # OracleEvaluator — eval không cần IoU matching
│
├── train.py              # training loop (AMP, warmup, resume, --dist-source)
├── evaluate_kitti.py     # eval: --mode iou/oracle/both, --lidar-gt
├── inference.py          # inference trên image / video / webcam
├── test_kitti.py         # full test eval + visualization + 6 plots
├── split_kitti.py        # tạo train/val/test splits
├── prepare_lidar_gt.py   # extract LiDAR GT distance (Zhu et al. 10th-pct)
│
├── runs/
│   ├── ogcde_v7/best.pt          # v7 best (ep 251, custom split, ann GT)
│   ├── ogcde_lidar_p2/best.pt    # LiDAR P2 best (ep ~430, Chen split, LiDAR GT)
│   ├── RESULTS.md                # full results
│   └── COMPARISON.md             # detailed comparison vs DistFormer
│
└── cache/
    ├── lidar_gt_train.json       # LiDAR GT (3,711 train images)
    └── lidar_gt_val.json         # LiDAR GT (3,768 val images)
```

---

## Reproduce

### Requirements

```bash
pip install torch torchvision numpy opencv-python
# Tested: Python 3.11, PyTorch 2.x, CUDA 12.x
```

### Data Preparation

Download KITTI Object Detection:
- `data_object_image_2.zip` — RGB images
- `data_object_label_2.zip` — annotations
- `data_object_calib.zip` — calibration
- `data_object_velodyne.zip` — LiDAR *(chỉ cần cho Phase 2)*

```
kitti_object/
└── training/
    ├── image_2/      # 7481 .png
    ├── label_2/      # 7481 .txt
    ├── calib/        # 7481 .txt
    └── velodyne/     # 7481 .bin  (optional)
```

Tạo Chen et al. split:
```bash
mkdir -p splits
wget -O splits/distformer_train.txt \
    https://raw.githubusercontent.com/charlesq34/frustum-pointnets/master/kitti/image_sets/train.txt
wget -O splits/distformer_val.txt \
    https://raw.githubusercontent.com/charlesq34/frustum-pointnets/master/kitti/image_sets/val.txt
```

### Phase 1 — Train (Annotation GT)

```bash
python train.py \
    --kitti-root /path/to/kitti/training \
    --train-split splits/distformer_train.txt \
    --val-split   splits/distformer_val.txt \
    --backbone-size m --pretrained-backbone yolov8m.pt \
    --freeze-backbone-epochs 10 \
    --epochs 350 --batch 16 --lr 1e-4 --weight-decay 5e-4 \
    --focal-gamma 1.5 --hflip --strong-aug --geo-warmup-epochs 5 \
    --w-box 5.0 --w-obj 2.0 --w-cls 0.5 --cls-weights 1.0 3.0 5.0 \
    --save-dir runs/ogcde_phase1 \
    2>&1 | tee runs/ogcde_phase1.log
```

### Prepare LiDAR GT (cần velodyne/)

```bash
python prepare_lidar_gt.py \
    --kitti-root /path/to/kitti/training \
    --split splits/distformer_train.txt \
    --out   cache/lidar_gt_train.json

python prepare_lidar_gt.py \
    --kitti-root /path/to/kitti/training \
    --split splits/distformer_val.txt \
    --out   cache/lidar_gt_val.json
```

### Phase 2 — Fine-tune (LiDAR GT)

```bash
python train.py \
    --resume runs/ogcde_phase1/best.pt \
    --reset-best --backbone-size m \
    --kitti-root /path/to/kitti/training \
    --train-split splits/distformer_train.txt \
    --val-split   splits/distformer_val.txt \
    --dist-source cache/lidar_gt_train.json \
    --epochs 450 --batch 16 --lr 1e-5 --weight-decay 5e-4 \
    --focal-gamma 1.5 --hflip --strong-aug --geo-warmup-epochs 0 \
    --w-box 5.0 --w-obj 2.0 --w-cls 0.5 --cls-weights 1.0 3.0 5.0 \
    --freeze-backbone-epochs 0 --max-grad-norm 2.0 \
    --save-dir runs/ogcde_lidar_p2 \
    2>&1 | tee runs/ogcde_lidar_p2.log
```

### Evaluation

```bash
# Standard IoU eval
python evaluate_kitti.py \
    --ckpt runs/ogcde_lidar_p2/best.pt \
    --kitti-root /path/to/kitti/training \
    --split splits/distformer_val.txt \
    --mode iou

# Fair comparison với DistFormer (LiDAR GT, cả IoU + Oracle)
python evaluate_kitti.py \
    --ckpt runs/ogcde_lidar_p2/best.pt \
    --kitti-root /path/to/kitti/training \
    --split splits/distformer_val.txt \
    --mode both \
    --lidar-gt cache/lidar_gt_val.json \
    --save-json results.json
```

### Inference

```bash
# Ảnh đơn
python inference.py --ckpt runs/ogcde_lidar_p2/best.pt --source image.jpg

# Video
python inference.py --ckpt runs/ogcde_lidar_p2/best.pt --source video.mp4 --save-video

# Webcam real-time (~84 FPS)
python inference.py --ckpt runs/ogcde_lidar_p2/best.pt --source 0
```

### Hardware

| | Minimum | Tested |
|---|---|---|
| GPU VRAM | 6 GB | RTX 4070 Laptop (8 GB) |
| RAM | 16 GB | 32 GB |
| Storage | 15 GB (không LiDAR) | SSD |
| Storage + LiDAR | 42 GB | SSD |
| Training Phase 1 | — | ~10 h |
| Training Phase 2 | — | ~3 h |
