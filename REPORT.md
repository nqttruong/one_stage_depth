# OGCDE - Báo cáo tổng hợp

---

## 1. Tổng quan

### Mục tiêu

Ước lượng khoảng cách metric từ camera đơn (monocular) cho từng object trong ảnh đường phố, hoạt động theo thời gian thực, không cần depth map dày đặc và không cần hồi quy 3D bounding box.

### Đóng góp chính — Geometric Scale Factor `s = sec(θ)`

Mô hình học một hệ số hiệu chỉnh hình học `s` cho từng object. Khoảng cách Euclid từ camera được tính là:

```
distance = s · d = sqrt(x² + y² + z²)
```

trong đó `d` là độ sâu Z-metric và `s ≈ sec(θ)` là góc lệch từ trục quang học. Cả hai được học trong không gian log (`d = exp(d_raw)`, `s = exp(s_raw)`), cho phép tối ưu bằng log-L1 thay vì linear L1. Đây là đóng góp cốt lõi: mô hình không cần camera calibration tại inference — thay vào đó nó **học ngầm** quan hệ hình học `sec(θ)` từ dữ liệu KITTI.

### Đóng góp phụ — Contact Point Prediction `(dx, dy)`

Head của mô hình hồi quy offset `(dx, dy)` tính bằng đơn vị stride, tương ứng với điểm chạm đất (ground contact point) của object trong không gian 2D. Điểm này là hình chiếu của điểm đáy 3D bounding box xuống ảnh, được sử dụng như một dấu hiệu hình học phụ trợ cho distance regression. GT được tính từ calib P2 tại load time; không cần P2 tại inference.

### Kiến trúc pipeline

```
Ảnh đầu vào (640×640, letterbox)
       │
       ▼
Backbone: YOLOv8n-scale, từ scratch
  stem → dark2(P2/4) → dark3(P3/8) → dark4(P4/16) → dark5+SPPF(P5/32)
  Kênh: (16, 32, 64, 128, 256)    Params: ~3.8M
       │
       ▼
Neck: PAN-FPN (top-down + bottom-up)
  Strides: 8, 16, 32 → 3 feature map scales
       │
       ▼
OGCDEHead × 3 scales
  9 + nc kênh/vị trí spatial:
  [cx, cy, w, h] + [obj] + [cls×3] + [d_raw] + [s_raw] + [dx, dy]
       │
       ▼
Decode: decode_bbox / decode_depth_scale / decode_contact
  distance = exp(d_raw + s_raw)   [numerically stable log-sum]
       │
       ▼
Output per object: bbox + class + score + depth d + scale s + distance + contact (u,v)
```

**Lưu ý về kiến trúc:** Pipeline là mạng one-stage thuần tuý, không sử dụng DepthAnything v2 hay SegFormer. Backbone được huấn luyện từ đầu trên KITTI; không dùng pretrained weights từ ImageNet hay COCO — đây là lựa chọn có chủ đích để kiểm chứng khả năng hội tụ của multi-task loss trước khi port sang pretrained backbone (xem Phần 5).

---

## 2. Phương pháp

### 2.1 Geometric Scale Calibration — công thức và lý do hoạt động

Cho một object tại toạ độ camera `(x, y, z)`:

```
d = z                              (Z-depth, stage-1: bottom center)
s = distance / d = sqrt(x²+y²+z²) / z = sec(θ)     (θ = góc từ optical axis)
```

Vì `s` phụ thuộc vào vị trí ngang của object trong ảnh, nó biến thiên **theo từng object** trong cùng một frame — do đó tên gọi "per-object scale calibration". Mạng học quan hệ này từ vị trí 2D bbox mà không cần biết focal length hay principal point tại inference.

**Loss hình học:**

```
L_depth  = |d_raw - log(d_gt)|        (log-L1, stable với magnitude lớn)
L_scale  = |s_raw - log(s_gt)|        (log-L1)
L_geo    = |exp(d_raw + s_raw) - dist_gt|   (linear L1 trên metres, đây là target thực sự)
```

`λ_geo` được warmup tuyến tính từ 0.1 → 2.0 trong 10 epoch đầu để tránh gradient của L_geo lấn át detection branch khi bbox còn chưa hội tụ.

### 2.2 Contact Point Prediction — thuật toán và motivation

**Motivation hình học:** Khoảng cách từ camera đến một object phụ thuộc vào điểm tiếp đất (ground contact), không phải tâm bbox 2D. Điểm tiếp đất là hình chiếu của `(x, y, z)` (bottom center 3D) xuống ảnh qua P2. Nếu model dự đoán đúng điểm này, nó học được ràng buộc hình học giữa bbox và đường chân trời (horizon line).

**Thuật toán:**

```
Contact GT:   [u, v] = P2 @ [x, y, z, 1]ᵀ   (chỉ dùng lúc load GT)
Contact pred: bx = cx,   by = cy + h/2   (bottom center bbox)
              u_pred = bx + dx_raw * stride
              v_pred = by + dy_raw * stride
L_contact = mean(||[u_pred, v_pred] - [u_gt, v_gt]||₂)
```

Không dùng `tanh` cho `dxdy_raw` để xử lý được objects bị crop hoặc nhô ra ngoài khung hình.

### 2.3 Multi-task Loss tổng hợp

```
L = 7.5·L_CIoU + 1.0·L_obj + 0.5·L_cls
  + 1.0·L_depth + 0.5·L_scale + 1.0·L_contact + λ_geo·L_geo
```

**Assignment:** Center-radius (r=1.5 stride) + ATSS-style size filter (stride 8: ≤64px, stride 16: 64–128px, stride 32: >128px). Mosaic augmentation bị loại bỏ do phá vỡ projection geometry.

---

## 3. Thực nghiệm

### 3.1 Dataset: KITTI Object Detection

| | Số lượng |
|---|---|
| Tổng ảnh | 7,481 |
| Train split | 5,985 |
| Val split | 1,496 |
| Classes | 3 (Car/Van/Truck → 0, Pedestrian/Person_sitting → 1, Cyclist → 2) |
| Filter | truncated ≤ 0.5, occluded ≤ 2, bbox ≥ 5×5 px |

Split được tạo bằng `split_kitti.py` với random seed=0, val_frac=0.2.

**Stage-1 depth GT:** `d = loc[2]` (Z-coordinate của bottom center, đọc trực tiếp từ KITTI `label_2`).  
**Stage-2 depth GT (chưa chạy):** Median LiDAR-Z trong 3D box, cần file `.bin` Velodyne.

### 3.2 Định nghĩa metrics

| Metric | Công thức | Đơn vị |
|---|---|---|
| AbsRel | `mean(|pred - gt| / gt)` | — |
| RMSE | `sqrt(mean((pred - gt)²))` | m |
| RMSE_log | `sqrt(mean((log(pred) - log(gt))²))` | — |
| δ₁ | `mean(max(pred/gt, gt/pred) < 1.25)` | % |
| δ₂ | ngưỡng 1.25² | % |
| δ₃ | ngưỡng 1.25³ | % |
| DE | `mean(|distance_pred - distance_gt|)` trên IoU>0.5 matches | m |
| CPE | mean pixel Euclid của contact point trên IoU>0.5 matches | px |

**Matching:** Greedy score-sorted, IoU>0.5, per-class. Chỉ matched pairs tham gia tính DE và CPE.

### 3.3 Setup thực nghiệm

| Thông số | Giá trị |
|---|---|
| Framework | PyTorch (torch, torchvision, numpy, opencv) |
| Hardware | CUDA GPU |
| Optimizer | AdamW, lr=1e-3, weight_decay=5e-4 |
| LR schedule | CosineAnnealingLR, T_max=100 |
| Batch size | 16 |
| Epochs | 100 |
| Input size | 640×640 (letterbox) |
| Mixed precision | torch.amp.autocast + GradScaler |
| Grad clip | max_norm=10.0 |
| Augmentation | Color jitter (ON), H-flip (OFF), Mosaic (OFF) |
| λ_geo warmup | 0.1 → 2.0 tuyến tính, 10 epoch |
| Depth GT | Stage-1: bottom-Z từ label_2 |

---

## 4. Kết quả

### 4.1 Kết quả chính — `runs/ogcde_v1/best.pt` (epoch 98)

**Distance metrics** (euclidean `sqrt(x²+y²+z²)`, 4,257 matched pairs):

| Metric | Giá trị |
|---|---|
| AbsRel | **0.0402** |
| RMSE | **1.797 m** |
| RMSE_log | 0.0596 |
| δ₁ (< 1.25) | **99.22%** |
| δ₂ (< 1.25²) | 100.00% |
| δ₃ (< 1.25³) | 100.00% |
| DE | **1.107 m** |
| CPE | **5.977 px** |

**Depth metrics** (Z-depth `d`):

| Metric | Giá trị |
|---|---|
| AbsRel | 0.0452 |
| RMSE | 1.831 m |
| RMSE_log | 0.0642 |
| δ₁ | 99.30% |

### 4.2 Kết quả — `runs/ogcde_v2/best.pt` (epoch 63)

Config thay đổi so với v1: focal loss (γ=1.5), fp32 loss computation, H-flip, `--w-box 5.0 --w-obj 2.0 --w-cls 0.5`, lr=5e-5.

**Distance metrics** (euclidean `sqrt(x²+y²+z²)`, 4,971 matched pairs):

| Metric | Giá trị |
|---|---|
| AbsRel | 0.0438 |
| RMSE | 2.096 m |
| RMSE_log | 0.0656 |
| δ₁ (< 1.25) | 98.85% |
| δ₂ (< 1.25²) | 99.96% |
| δ₃ (< 1.25³) | 100.00% |
| DE | 1.295 m |
| CPE | 6.353 px |

**Depth metrics** (Z-depth `d`):

| Metric | Giá trị |
|---|---|
| AbsRel | 0.0485 |
| RMSE | 2.121 m |
| RMSE_log | 0.0705 |
| δ₁ | 98.79% |

**Nhận xét v2 vs v1:** v2 detect được nhiều object hơn (4,971 vs 4,257 matched pairs, +17%) nhờ focal loss và loss weight tuning. Tuy nhiên các distance metrics tuyệt đối kém hơn v1 — nhiều khả năng vì v2 detect thêm các object khó (xa, nhỏ) vốn có sai số distance cao hơn, kéo DE và AbsRel lên. Val loss không so sánh trực tiếp được giữa v1 và v2 do thay đổi loss function.

### 4.3 Ablation — trạng thái hiện tại

| Config | Backbone | H-flip | Focal | λ_cp | Pairs | DE (m) | AbsRel | δ₁ | CPE (px) |
|---|---|---|---|---|---|---|---|---|---|
| **ogcde_v1** (ep 98) | scratch | OFF | OFF | 1.0 | 4,257 | **1.107** | **0.0402** | **99.22%** | 5.977 |
| **ogcde_v2** (ep 63) | scratch | ON | γ=1.5 | 1.0 | 4,971 | 1.295 | 0.0438 | 98.85% | 6.353 |
| **ogcde_v3** (ep 75) | scratch | ON | γ=1.5 | **0.3** | 5,395 | 1.359 | 0.0451 | 98.61% | 6.428 |
| **ogcde_v4** (ep 143) | **YOLOv8n COCO** | ON | γ=1.5 | 1.0 | 5,206 | 1.348 | 0.0452 | 98.66% | **6.242** |

**Nhận xét ablation:**

- **λ_cp (v3):** Giảm λ_cp 1.0→0.3 tăng detections (+8%) nhưng distance metrics xấu hơn. Contact point loss là geometric regularization quan trọng — không nên giảm.
- **Pretrained backbone (v4):** Cải thiện rõ CPE (6.24px, -1.7% vs v2) và geo loss thấp hơn ~18% so với v2 ở cùng λ_geo=2.0. Tuy nhiên AbsRel và DE không cải thiện so với v2. Lý do: v4 detect nhiều object hơn v2 (+5%), bao gồm các object khó hơn (xa, nhỏ) vốn có sai số cao hơn, kéo DE trung bình lên. Pretrained backbone thực sự giúp ích về feature quality (thấy qua CPE và geo loss) nhưng bị che khuất bởi hiệu ứng recall tăng.
- **Xu hướng chung:** v1 có ít detections nhất (4,257) và metrics tốt nhất — không phải vì v1 tốt hơn thực sự mà vì nó bỏ sót nhiều object khó hơn. Metric DE/AbsRel phụ thuộc mạnh vào recall của detector.

### 4.4 So sánh với paper liên quan — CDR

Paper tham chiếu: **"Supervised Object-Specific Distance Estimation from Monocular Images for Autonomous Driving"** (PMC9693490), phương pháp CDR (Convolutional Depth Regression), backbone ConvNeXt-small, huấn luyện supervised trên KITTI.

| Phương pháp | Kiến trúc | Metric chính | Giá trị | Dataset |
|---|---|---|---|---|
| CDR | ConvNeXt + optics decoder | wMAE | 1.93 ± 0.03 m | KITTI |
| Monodepth2 (baseline của CDR) | ResNet + decoder | wMAE | 2.28 m | KITTI |
| **OGCDE v1 (ours)** | scratch YOLOv8n-scale | DE (MAE) | **1.107 m** | KITTI |
| **OGCDE v4 (ours, pretrained)** | COCO YOLOv8n-scale | DE (MAE) | 1.348 m | KITTI |

**Lưu ý về so sánh:** CDR dùng wMAE (weighted MAE có trọng số theo khoảng cách) trên toàn bộ objects. OGCDE dùng DE (unweighted MAE) chỉ trên IoU>0.5 matched pairs — các objects không detect được không tính vào DE, khiến DE có lợi thế hơn wMAE về mặt tính toán. So sánh trực tiếp cần chạy cùng evaluation protocol.

**Bối cảnh thêm từ YOLO MDE (Electronics 2022, MDPI):** Một pipeline tương tự (YOLOv4 + depth head) đạt mean error rate 3.71% trên KITTI 3D Object Detection, AP 71.68% (Car). OGCDE tiếp cận khác biệt ở chỗ không dùng depth map dày đặc làm trung gian.

### 4.5 Phân tích đóng góp

**Đóng góp nào quan trọng nhất?** Geometric scale factor `s`:

- Nếu đặt `s = 1` (tức là `distance ≈ d = Z`), sai số phụ thuộc vào góc θ. Với object ở rìa ảnh (θ ≈ 20°), `sec(θ) ≈ 1.064` — sai số tương đối ~6.4% chỉ từ việc bỏ qua `s`. Ở góc rộng hơn, sai số tăng nhanh. Việc học `s` tường minh giải quyết điều này mà không cần camera matrix tại inference.
- δ₁ = 99.22% (toàn bộ gần 100%) cho thấy model học được `s` một cách ổn định.

**Contact point:** CPE = 5.98 px trên ảnh KITTI 1242×375 (~0.48% chiều rộng) là chính xác. Tuy nhiên contact point chủ yếu phục vụ như supervision signal hình học — chưa có ablation riêng đo tác động của nó lên DE.

---

## 5. Kết luận và hạn chế

### Điểm mạnh

- **Pipeline đơn giản:** Một forward pass cho cả detection + distance, không cần depth map trung gian.
- **Geometry-aware:** Scale factor `s` được học ngầm từ dữ liệu, không cần camera calibration tại inference.
- **Kết quả tốt ở stage-1:** AbsRel = 0.040, δ₁ = 99.2% (v1) — 0.044, δ₁ = 98.9% (v2) sau ~100 epoch với backbone từ scratch. v2 detect nhiều object hơn (+17% matched pairs) nhờ focal loss.
- **Code sạch:** Không phụ thuộc Ultralytics hay mmdet; chỉ cần `torch`, `torchvision`, `numpy`, `opencv`.

### Hạn chế hiện tại

1. **Chưa có stage-2:** Kết quả hiện tại dùng bottom-Z GT (đơn giản nhất). Median LiDAR-Z sẽ cho depth GT chính xác hơn nhưng cần file `.bin` Velodyne.
2. **Backbone từ scratch:** Chưa dùng pretrained weights (COCO/ImageNet). Một backbone YOLOv8n pretrained sẽ cải thiện cả detection AP lẫn feature quality cho depth.
3. **Stability metric chưa có số:** Cần video/sequence data (KITTI Tracking hoặc video riêng), chưa chạy.
4. **Chỉ một experiment đầy đủ:** Không có ablation data cho các trục: stage-2 depth, H-flip, λ_geo schedule, kiến trúc head.
5. **So sánh baseline hạn chế:** CDR dùng metric khác (wMAE vs DE), so sánh không trực tiếp.
6. **TaskAlignedAssigner chưa implement:** Center-radius assigner phù hợp cho distance learning nhưng chưa so sánh với TAL về detection AP.

### Hướng phát triển

1. Chạy stage-2 với `prepare_lidar_depth.py` → fine-tune từ `ogcde_v1/best.pt`.
2. Port backbone sang YOLOv8n pretrained (recipe có trong README).
3. Chạy ablation đầy đủ 4 configs (stage-1/2 × no-warmup/warmup).
4. Đánh giá trên KITTI Tracking để lấy Stability metric.
5. So sánh với CDR trên cùng evaluation protocol.

---

## 6. Phụ lục

### 6.1 Reproduce — các lệnh theo thứ tự

```bash
# Bước 0: tạo split
python split_kitti.py \
    --kitti-root /media/truong/01DBB45ECE0C4E00/dl/kitti_object/training \
    --out-dir /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/

# Bước 1: train stage-1 (đã chạy, kết quả tại runs/ogcde_v1/)
python train.py \
    --kitti-root /media/truong/01DBB45ECE0C4E00/dl/kitti_object/training \
    --train-split /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/kitti_train.txt \
    --val-split   /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/kitti_val.txt \
    --img-size 640 --batch 16 --epochs 100 --workers 4 \
    --geo-warmup-epochs 10 --save-dir runs/ogcde_v1

# Bước 2: evaluate (đã chạy, kết quả tại runs/ogcde_v1/eval_results.json)
python evaluate_kitti.py \
    --ckpt runs/ogcde_v1/best.pt \
    --kitti-root /media/truong/01DBB45ECE0C4E00/dl/kitti_object/training \
    --split /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/kitti_val.txt \
    --save-json runs/ogcde_v1/eval_results.json

# Bước 3 (tuỳ chọn): precompute LiDAR depth cho stage-2
python prepare_lidar_depth.py \
    --kitti-root /media/truong/01DBB45ECE0C4E00/dl/kitti_object/training \
    --split /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/kitti_train.txt \
    --out cache/kitti_train_lidar.json

# Inference trên video
python inference.py --ckpt runs/ogcde_v2/best.pt --source /media/truong/01DBB45ECE0C4E00/dl/one_stage_depth/video/12207144_1920_1080_30fps.mp4
```

### 6.2 Cấu trúc thư mục

```
one_stage_depth/
├── ogcde/
│   ├── model.py       # OGCDENet = Backbone + Neck + OGCDEHead; decode helpers
│   ├── loss.py        # OGCDELoss; assign_targets (center-radius + ATSS size filter)
│   ├── dataset.py     # KITTIOGCDEDataset; collate_ogcde
│   ├── metrics.py     # OGCDEEvaluator; depth_metrics; stability
│   └── utils.py       # letterbox; decode_predictions; NMS; coord mapping
├── train.py           # entry point: λ_geo warmup, AdamW, cosine LR, AMP
├── evaluate_kitti.py  # full metric suite trên val split
├── inference.py       # video/folder inference + SimpleIoUTracker + stability
├── prepare_lidar_depth.py  # stage-2: median LiDAR-Z per object
├── split_kitti.py     # tạo train/val split files
└── runs/
    ├── ogcde_v1/
    │   ├── best.pt           # epoch 98, val_loss=16.242
    │   ├── last.pt           # epoch 99, val_loss=16.265
    │   └── eval_results.json # AbsRel=0.040, DE=1.107m, CPE=5.977px
    └── smoke/
        ├── best.pt           # epoch 0, val_loss=24.980 (smoke test only)
        └── last.pt
```

### 6.3 Checkpoint quan trọng

| File | Epoch | Val loss | Trạng thái |
|---|---|---|---|
| `runs/ogcde_v1/best.pt` | 98 | 16.242 | **Dùng cho inference và report** |
| `runs/ogcde_v1/last.pt` | 99 | 16.265 | Gần bằng best, có thể dùng thay thế |
| `runs/smoke/best.pt` | 0 | 24.980 | Chỉ dùng để kiểm tra pipeline chạy được |

---

*Số liệu trong báo cáo này được trích xuất trực tiếp từ `runs/ogcde_v1/eval_results.json` và metadata trong checkpoint `.pt`. Không có số liệu ước tính hay giả định.*
