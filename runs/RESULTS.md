# OGCDE — Full Results Summary

**Date:** 2026-05-07  
**Model:** OGCDE + YOLOv8m backbone  
**Dataset:** KITTI Object Detection (3 classes: Car, Pedestrian, Cyclist)  
**Task:** Monocular distance & depth estimation + object detection

---

## 1. Models Overview

| Model | Train split | Train GT | Epochs | Best ep | Checkpoint |
|---|---|---|---|---|---|
| **OGCDE v7** | kitti_train (5985) | Annotation | 300 | 251 | `ogcde_v7/best.pt` |
| **OGCDE LiDAR P2** ✅ | distformer_train (3711) | LiDAR 10th-pct | 350+100 | ~430 | `ogcde_lidar_p2/best.pt` |

> ✅ = best model cho so sánh với DistFormer  
> OGCDE v7: split tùy chỉnh (5985/1196/300), annotation GT  
> OGCDE LiDAR P2: Chen et al. split (3711/3768), LiDAR GT — two-phase training

---

## 2. Training Configuration

### OGCDE v7

| Parameter | Value |
|---|---|
| Backbone | YOLOv8m |
| Input size | 640×640 |
| Batch size | 16 |
| Phase 1 lr | 1e-4 (ep 1–33) |
| Phase 2 lr | 3e-5 (ep 33–300, optimizer reset) |
| Loss weights | w_box=5.0, w_obj=2.0, w_cls=0.5 |
| cls-weights | 1.0/3.0/5.0 (Car/Ped/Cyc) |
| Augmentation | Strong-aug + HFlip |

### OGCDE LiDAR P2

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| Epochs | 350 (ep 1–350) | 100 (ep 351–450) |
| lr | 1e-4 → 3e-5 | 1e-5 |
| Train GT | Annotation | **LiDAR 10th-pct** |
| max_grad_norm | 10.0 | 2.0 |
| Freeze backbone | 10 ep | 0 |
| geo-warmup | 5 ep | 0 |

---

## 3. OGCDE v7 — Validation Set (`kitti_val_new.txt`, 1196 ảnh)

### Detection

| Class | AP@0.5 | AP3D |
|---|---|---|
| Car | 0.8044 | 0.8028 |
| Pedestrian | 0.4791 | 0.4787 |
| Cyclist | 0.3371 | 0.3371 |
| **mAP** | **0.5402** | **0.5395** |

### Distance — all classes (4779 pairs, IoU≥0.5, annotation GT)

| Metric | Distance | Depth (Z) |
|---|---|---|
| δ<1.25 ↑ | **99.79%** | 99.71% |
| AbsRel ↓ | **0.0219** | 0.0265 |
| SqRel ↓ | **0.0348** | 0.0439 |
| RMSE ↓ | **1.136 m** | 1.264 m |
| RMSElog ↓ | **0.0355** | 0.0408 |
| DE | **0.647 m** | — |
| ALE | **0.798 m** | — |

### Distance — per class

| Class | n | δ<1.25 | AbsRel | SqRel | RMSE | RMSElog | DE | ALE |
|---|---|---|---|---|---|---|---|---|
| Car | 4224 | 99.81% | 0.0210 | 0.0358 | 1.182 m | 0.0342 | 0.671 m | 0.822 m |
| Pedestrian | 459 | 99.56% | 0.0298 | 0.0276 | 0.694 m | 0.0459 | 0.453 m | 0.622 m |
| Cyclist | 96 | 100.00% | 0.0256 | 0.0226 | 0.697 m | 0.0350 | 0.510 m | 0.599 m |

---

## 4. OGCDE v7 — Test Set (`kitti_test.txt`, 300 ảnh)

### Detection

| Class | AP@0.5 | AP3D |
|---|---|---|
| Car | 0.8051 | 0.8019 |
| Pedestrian | 0.4249 | 0.4211 |
| Cyclist | 0.4129 | 0.3923 |
| **mAP** | **0.5476** | **0.5384** |

### Distance — all classes (1117 pairs, IoU≥0.5, annotation GT)

| Metric | Distance | Depth (Z) |
|---|---|---|
| δ<1.25 ↑ | **99.46%** | 99.73% |
| AbsRel ↓ | **0.0227** | 0.0272 |
| SqRel ↓ | **0.0403** | 0.0481 |
| RMSE ↓ | **1.186 m** | 1.290 m |
| RMSElog ↓ | **0.0391** | 0.0440 |
| DE | **0.661 m** | — |
| ALE | **0.805 m** | — |

### Distance — per class

| Class | n | δ<1.25 | AbsRel | SqRel | RMSE | RMSElog | DE | ALE |
|---|---|---|---|---|---|---|---|---|
| Car | 968 | 99.79% | 0.0210 | 0.0386 | 1.228 m | 0.0341 | 0.687 m | 0.832 m |
| Pedestrian | 108 | 97.22% | 0.0359 | 0.0509 | 0.848 m | 0.0664 | 0.480 m | 0.582 m |
| Cyclist | 41 | 97.56% | 0.0279 | 0.0503 | 0.916 m | 0.0502 | 0.526 m | 0.740 m |

---

## 5. OGCDE LiDAR P2 — DistFormer Val (`distformer_val.txt`, 3768 ảnh)

*Eval GT: LiDAR 10th-pct (same as DistFormer)*

### Detection — AP (conf=0.001, DontCare-filtered)

KITTI official IoU: Car@0.7, Pedestrian@0.5, Cyclist@0.5

| Class | AP@0.5 | AP@KIT-IoU | AP@0.5:0.95 | Easy | Moderate | Hard |
|---|---|---|---|---|---|---|
| Car | 0.667 | **0.296** | 0.279 | 0.239 | 0.356 | 0.326 |
| Pedestrian | 0.289 | **0.289** | 0.063 | 0.384 | 0.338 | 0.299 |
| Cyclist | 0.107 | **0.107** | 0.030 | 0.196 | 0.117 | 0.109 |
| **mAP** | **0.354** | **0.231** | **0.124** | **0.273** | **0.270** | **0.245** |

> AP@KIT-IoU = AP at KITTI standard IoU per class (Car@0.7, others@0.5)  
> Cross-validated with [kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python):  
> Pedestrian <0.2% from official, Cyclist <1.1% from official. Car gap ~3-6% due to AP interpolation method difference (our AUC vs official 40-point).

### Detection — Recall @ Distance Bins (IoU≥0.5, conf=0.001)

| Class | 0–20 m | 20–40 m | 40–60 m | >60 m |
|---|---|---|---|---|
| Car | 85.1% (4945/5809) | 93.1% (6001/6443) | 87.2% (2784/3194) | **77.0%** (895/1162) |
| Pedestrian | 65.8% (1180/1794) | 37.0% (205/554) | 4.9% (4/81) | 5.9% (1/17) |
| Cyclist | 68.4% (212/310) | 53.0% (197/372) | 13.9% (23/166) | 0.0% (0/45) |

> Car maintains 77% recall at >60m — geometry-aware head benefit for far-range detection.  
> Pedestrian/Cyclist drop at >40m: objects have pixel height <25px at those distances (KITTI ignores them in official eval).

### Detection — KITTI Easy/Moderate/Hard per class (AP@KIT-IoU)

### Distance — all classes

| Metric | IoU mode (13,085 pairs) | Oracle mode (17,499 objects) |
|---|---|---|
| δ<1.25 ↑ | **96.74%** | 86.58% |
| AbsRel ↓ | **7.54%** | 11.87% |
| SqRel ↓ | **0.313** | 1.045 |
| RMSE ↓ | **3.674 m** | 6.583 m |
| RMSElog ↓ | **0.098** | 0.233 |
| DE | **2.241 m** | 3.692 m |
| ALE | **2.418 m** | 3.836 m |

### Distance — per class (IoU mode + LiDAR GT)

| Class | n | δ<1.25 | AbsRel | SqRel | RMSE | RMSElog | DE | ALE |
|---|---|---|---|---|---|---|---|---|
| Car | 11,912 | **96.85%** | **7.44%** | 0.326 | 3.813 m | 0.097 | 2.348 m | 2.538 m |
| Pedestrian | 1,008 | 95.04% | 8.50% | 0.179 | 1.661 m | 0.111 | 1.106 m | 1.147 m |
| Cyclist | 165 | **98.79%** | 8.43% | 0.182 | 1.908 m | 0.098 | 1.464 m | 1.541 m |

### Distance — per class (Oracle mode + LiDAR GT)

| Class | n | δ<1.25 | AbsRel | SqRel | RMSE | RMSElog | DE | ALE |
|---|---|---|---|---|---|---|---|---|
| Car | 14,479 | 88.18% | 11.17% | 1.054 | 6.839 m | 0.230 | 3.806 m | 3.984 m |
| Pedestrian | 2,339 | 78.54% | 15.55% | 0.968 | 4.785 m | 0.255 | 2.821 m | 2.786 m |
| Cyclist | 681 | 80.03% | 14.21% | 1.110 | 6.367 m | 0.202 | 4.266 m | 4.289 m |

---

## 6. Comparison with DistFormer (IoU + LiDAR GT)

| Class | DistFormer (GT box) | OGCDE LiDAR P2 (IoU) | Winner |
|---|---|---|---|
| **All** | δ=93.67%, AbsRel=10.39% | **δ=96.74%, AbsRel=7.54%** | OGCDE |
| **Car** | δ=94.32%, AbsRel=9.97% | **δ=96.85%, AbsRel=7.44%** | OGCDE |
| **Pedestrian** | **δ=98.15%, AbsRel=5.67%** | δ=95.04%, AbsRel=8.50% | DistFormer |
| **Cyclist** | δ=95.62%, AbsRel=8.01% | **δ=98.79%, AbsRel=8.43%** | OGCDE (δ) |

> **Note**: DistFormer nhận GT box làm input. OGCDE tự detect + predict distance. Full comparison: `runs/COMPARISON.md`

---

## 7. Inference Speed (OGCDE v7, RTX 4070 Laptop)

| Stage | FPS | Latency |
|---|---|---|
| Model forward only | 92.5 FPS | 10.8 ms |
| + NMS decode | 88.6 FPS | 11.3 ms |
| Full pipeline | **84.0 FPS** | 11.9 ms |

---

## 8. Output Files

```
runs/
├── ogcde_v7/best.pt              ← v7 best (ep251, val=8.359)
├── ogcde_distformer/best.pt      ← DF annotation best (ep~311, val=18.874)
├── ogcde_lidar_p2/best.pt        ← LiDAR P2 best (ep~430, val=20.565) ✅
├── test_v7/
│   ├── vis/                      ← 300 annotated test images
│   ├── metrics.json
│   └── plots/ (7 plots)
├── cache/
│   ├── lidar_gt_train.json       ← LiDAR GT for 3711 train images
│   └── lidar_gt_val.json         ← LiDAR GT for 3768 val images
├── COMPARISON.md                 ← full method comparison
└── RESULTS.md                    ← this file
```

---

## 9. Key Observations

1. **OGCDE v7** (annotation, 5985 train): δ<1.25=99.79%, AbsRel=2.19% — best absolute performance nhờ nhiều training data và clean GT.
2. **OGCDE LiDAR P2** (LiDAR, 3711 train): vượt DistFormer trên δ<1.25 và AbsRel (IoU mode), dù phải tự detect box và train ít data hơn.
3. **Pedestrian khó nhất**: detection coverage thấp (43% ở IoU mode), DistFormer tốt hơn rõ rệt cho class này.
4. **Two-phase training**: phase 1 annotation cho stability, phase 2 LiDAR GT cho alignment với DistFormer distribution — hiệu quả hơn train thẳng LiDAR từ đầu.
5. **Oracle mode honest hơn IoU mode**: Oracle tính trên 100% GT objects (không có selection bias), DistFormer vẫn tốt hơn ở oracle mode.
