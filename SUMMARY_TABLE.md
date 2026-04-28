# Bảng kết quả OGCDE — dùng để paste vào slide

## Kết quả chính (KITTI val, 4,257 matched pairs, IoU>0.5)

| Metric | Distance | Depth (Z) |
|---|---|---|
| AbsRel ↓ | **0.0402** | 0.0452 |
| RMSE (m) ↓ | **1.797** | 1.831 |
| RMSE_log ↓ | **0.0596** | 0.0642 |
| δ₁ < 1.25 ↑ | **99.22%** | 99.30% |
| δ₂ < 1.25² ↑ | 100.00% | 100.00% |
| δ₃ < 1.25³ ↑ | 100.00% | 100.00% |
| DE (m) ↓ | **1.107** | — |
| CPE (px) ↓ | **5.977** | — |

> Checkpoint: `runs/ogcde_v1/best.pt` (epoch 98)  
> Config: stage-1, bottom-Z GT, batch=16, 100 epochs, img=640

---

## So sánh với CDR baseline

| Phương pháp | Backbone | Metric | Giá trị | Ghi chú |
|---|---|---|---|---|
| Monodepth2 | ResNet | wMAE | 2.28 m | Self-supervised |
| CDR | ConvNeXt-small | wMAE | 1.93 ± 0.03 m | 15% tốt hơn Monodepth2 |
| **OGCDE v1 (ours)** | YOLOv8n-scale | DE (MAE) | **1.107 m** | Stage-1, IoU>0.5 matched |

> ⚠️ CDR dùng wMAE trên tất cả objects; OGCDE dùng DE chỉ trên detected+matched objects.  
> So sánh trực tiếp cần cùng evaluation protocol.

---

## Trạng thái ablation

| Config | Val loss | DE (m) | AbsRel | Trạng thái |
|---|---|---|---|---|
| stage-1, bottom-Z, no hflip | **16.242** | **1.107** | **0.0402** | ✅ Đã chạy |
| stage-2, LiDAR-Z | — | — | — | ⏳ Chưa chạy |
| + H-flip | — | — | — | ⏳ Chưa chạy |
| no λ_geo warmup | — | — | — | ⏳ Chưa chạy |

---

## Thông số mô hình

| | |
|---|---|
| Kiến trúc | YOLOv8n-scale + PAN-FPN + OGCDEHead |
| Số parameters | **3,825,735** (~3.8M) |
| Input | 640×640, letterbox |
| Classes | 3 (Car, Pedestrian, Cyclist) |
| Strides | 8 / 16 / 32 |
| Head channels/loc | 9 + nc = 12 |
| Pretrained | Không (từ scratch) |
| Training data | KITTI 5,985 ảnh |
| Val data | KITTI 1,496 ảnh |
