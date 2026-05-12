# OGCDE — Full Method Comparison & DistFormer

**Date:** 2026-05-07

---

## 1. Mô tả các phương pháp

| ID | Tên | Train data | Epochs | Best ep | Train GT | Eval GT | Eval mode |
|---|---|---|---|---|---|---|---|
| **V7-val** | OGCDE v7 | 5985 ảnh (kitti_train.txt) | 300 | 251 | Annotation | Annotation | IoU≥0.5 |
| **V7-test** | OGCDE v7 | 5985 ảnh (kitti_train.txt) | 300 | 251 | Annotation | Annotation | IoU≥0.5 |
| **DF-ann** | OGCDE DF split | 3711 ảnh (distformer_train.txt) | 200 | 1* | Annotation | Annotation/LiDAR | IoU/Oracle |
| **DF-R** | OGCDE DF split (resume) | 3711 ảnh | 350 | ~311 | Annotation | LiDAR | IoU/Oracle |
| **LID-p1** | OGCDE LiDAR train | 3711 ảnh | ~280† | ~280† | **LiDAR** | **LiDAR** | IoU/Oracle |
| **LID-P2** | OGCDE LiDAR phase2 ✅ | 3711 ảnh | 350+100 | ~430 | **LiDAR** | **LiDAR** | IoU/Oracle |
| **DistFormer** | DistFormer | 3712 ảnh (Chen et al.) | — | — | LiDAR | LiDAR | GT box |

> ✅ = best model cho so sánh với DistFormer  
> \* Model chưa converge (best ep 1)  
> † Dừng sớm do NaN gradient từ ep 387+  
> **LID-P2**: Phase 1 = annotation GT (350 ep, lr=1e-4→3e-5), Phase 2 = fine-tune LiDAR GT (100 ep, lr=1e-5, max_grad_norm=2.0)

### Backbone & Training config (tất cả OGCDE variants)

| Parameter | Value |
|---|---|
| Backbone | YOLOv8m (pretrained COCO) |
| Input size | 640×640 |
| Batch size | 16 |
| Optimizer | AdamW, weight-decay=5e-4 |
| Loss weights | w_box=5.0, w_obj=2.0, w_cls=0.5 |
| cls-weights | 1.0/3.0/5.0 (Car/Ped/Cyc) |
| Augmentation | Strong-aug + HFlip |

### Giải thích eval mode

- **IoU≥0.5**: model tự detect box → match GT qua IoU≥0.5 → tính metric trên matched pairs (~74-79% GT objects)
- **Oracle**: với mỗi GT box, query prediction tại center cell trên feature grid → tính trên 100% GT objects
- **GT box** (DistFormer): model nhận GT box làm input, predict distance trên 100% GT objects

### Giải thích GT distance

- **Annotation**: `sqrt(X²+Y²+Z²)` từ `loc` field trong KITTI label — chính xác, không noise
- **LiDAR 10th-pct**: 10th percentile của LiDAR points trong 3D bounding box (Zhu et al., DistFormer method)

---

## 2. Detection — Overview (mAP@IoU≥0.5)

| ID | Val/Test | Car AP | Ped AP | Cyc AP | mAP | mAP3D |
|---|---|---|---|---|---|---|
| V7-val | kitti_val_new (1196) | 0.8044 | 0.4791 | 0.3371 | **0.5402** | 0.5395 |
| V7-test | kitti_test (300) | 0.8051 | 0.4249 | 0.4129 | **0.5476** | 0.5384 |
| DF-R | distformer_val (3768) | 0.6771 | 0.2609 | 0.0580 | 0.3320 | 0.2793 |
| LID-p1 | distformer_val (3768) | 0.6696 | 0.2678 | 0.0568 | 0.3314 | 0.3116 |
| **LID-P2** | distformer_val (3768) | **0.6758** | **0.2690** | **0.0613** | **0.3353** | **0.3222** |

## 2b. Detection — LID-P2 Full (KITTI-standard, conf=0.001, DontCare-filtered)

*KITTI IoU threshold: Car@0.7, Pedestrian@0.5, Cyclist@0.5*

| Class | AP@0.5 | AP@KIT-IoU | AP@0.5:0.95 | Easy | Moderate | Hard |
|---|---|---|---|---|---|---|
| Car | 0.667 | **0.296** | 0.279 | 0.239 | 0.356 | 0.326 |
| Pedestrian | 0.289 | **0.289** | 0.063 | 0.384 | 0.338 | 0.299 |
| Cyclist | 0.107 | **0.107** | 0.030 | 0.196 | 0.117 | 0.109 |
| **mAP** | **0.354** | **0.231** | **0.124** | **0.273** | **0.270** | **0.245** |

> Cross-validated với [kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python):  
> Ped gap <0.2%, Cyclist gap <1.1%. Car gap ~3–6% do khác interpolation method (AUC vs 40-point).

## 2c. Detection — Recall @ Distance Bins (LID-P2, IoU≥0.5, conf=0.001)

| Class | 0–20 m | 20–40 m | 40–60 m | >60 m |
|---|---|---|---|---|
| Car | 85.1% (4945/5809) | 93.1% (6001/6443) | 87.2% (2784/3194) | **77.0%** (895/1162) |
| Pedestrian | 65.8% (1180/1794) | 37.0% (205/554) | 4.9% (4/81) | 5.9% (1/17) |
| Cyclist | 68.4% (212/310) | 53.0% (197/372) | 13.9% (23/166) | 0.0% (0/45) |

> Car 77% recall at >60m — geometry-aware head advantage for far-range detection.  
> Ped/Cyc drop at >40m: object height <25px at those distances (KITTI officially ignores them).

---

## 3. Distance metrics — All classes

| ID | Mode | n | δ<1.25↑ | AbsRel↓ | SqRel↓ | RMSE↓ | RMSElog↓ | DE↓ | ALE↓ |
|---|---|---|---|---|---|---|---|---|---|
| V7-val | IoU | 4779 | **99.79%** | **0.0219** | **0.0348** | **1.136 m** | **0.0355** | **0.647 m** | **0.798 m** |
| V7-test | IoU | 1117 | 99.46% | 0.0227 | 0.0403 | 1.186 m | 0.0391 | 0.661 m | 0.805 m |
| DF-ann | IoU+Ann | 13238 | 96.31% | 0.0733 | 0.3538 | 4.153 m | 0.0994 | 2.439 m | 2.525 m |
| DF-ann | Oracle+Ann | 17499 | 86.98% | 0.1122 | 1.0010 | 6.693 m | 0.2117 | 3.769 m | 3.844 m |
| DF-R | IoU+LiDAR | 13230 | 89.97% | 0.1256 | 0.4726 | 3.849 m | 0.1415 | 2.902 m | 2.308 m |
| DF-R | Oracle+LiDAR | 17499 | 81.40% | 0.1565 | 1.0829 | 6.359 m | 0.2293 | 4.102 m | 3.742 m |
| LID-p1 | IoU+LiDAR | 13902 | 95.10% | 0.0859 | 0.3792 | 3.954 m | 0.1104 | 2.491 m | 2.591 m |
| LID-p1 | Oracle+LiDAR | 17499 | 86.34% | 0.1231 | 1.0121 | 6.337 m | 0.2225 | 3.681 m | 3.794 m |
| **LID-P2** | **IoU+LiDAR** | **13085** | **96.74%** | **0.0754** | **0.3131** | **3.674 m** | **0.0984** | **2.241 m** | **2.418 m** |
| **LID-P2** | **Oracle+LiDAR** | **17499** | **86.58%** | **0.1187** | **1.0448** | **6.583 m** | **0.2328** | **3.692 m** | **3.836 m** |
| DistFormer | GT box+LiDAR | all GT | 93.67% | 0.1039 | 0.32 | **2.950 m** | 0.150 | — | — |

---

## 4. Distance metrics — Class: Car

| ID | Mode | n | δ<1.25↑ | AbsRel↓ | SqRel↓ | RMSE↓ | RMSElog↓ | DE↓ | ALE↓ |
|---|---|---|---|---|---|---|---|---|---|
| V7-val | IoU | 4224 | **99.81%** | **0.0210** | **0.0358** | **1.182 m** | **0.0342** | **0.671 m** | **0.822 m** |
| DF-R | IoU+LiDAR | 12039 | 89.49% | 0.1285 | 0.4987 | 3.995 m | 0.1436 | 3.062 m | 2.421 m |
| DF-R | Oracle+LiDAR | 14479 | 81.94% | 0.1558 | 1.0946 | 6.564 m | 0.2290 | 4.279 m | 3.861 m |
| LID-p1 | IoU+LiDAR | 12552 | 95.18% | 0.0849 | 0.3969 | 4.118 m | 0.1094 | 2.621 m | 2.735 m |
| LID-p1 | Oracle+LiDAR | 14479 | 87.89% | 0.1161 | 1.0070 | 6.547 m | 0.2179 | 3.792 m | 3.937 m |
| **LID-P2** | **IoU+LiDAR** | **11912** | **96.85%** | **0.0744** | **0.3263** | **3.813 m** | **0.0972** | **2.348 m** | **2.538 m** |
| **LID-P2** | **Oracle+LiDAR** | **14479** | **88.18%** | **0.1117** | **1.0541** | **6.839 m** | **0.2304** | **3.806 m** | **3.984 m** |
| DistFormer | GT box+LiDAR | all GT | 94.32% | 0.0997 | **0.22** | **2.110 m** | **0.130** | — | — |

---

## 5. Distance metrics — Class: Pedestrian

| ID | Mode | n | δ<1.25↑ | AbsRel↓ | SqRel↓ | RMSE↓ | RMSElog↓ | DE↓ | ALE↓ |
|---|---|---|---|---|---|---|---|---|---|
| V7-val | IoU | 459 | **99.56%** | **0.0298** | **0.0276** | **0.694 m** | **0.0459** | **0.453 m** | **0.622 m** |
| DF-R | IoU+LiDAR | 1068 | 94.57% | 0.0976 | 0.2117 | 1.806 m | 0.1192 | 1.269 m | 1.145 m |
| DF-R | Oracle+LiDAR | 2339 | 78.54% | 0.1604 | 0.9419 | 4.710 m | 0.2378 | 2.847 m | 2.760 m |
| LID-p1 | IoU+LiDAR | 1148 | 93.73% | 0.0992 | 0.2191 | 1.758 m | 0.1233 | 1.227 m | 1.152 m |
| LID-p1 | Oracle+LiDAR | 2339 | 78.84% | 0.1570 | 0.8989 | 4.508 m | 0.2485 | 2.717 m | 2.654 m |
| **LID-P2** | **IoU+LiDAR** | **1008** | **95.04%** | **0.0850** | **0.1790** | **1.661 m** | **0.1110** | **1.106 m** | **1.147 m** |
| **LID-P2** | **Oracle+LiDAR** | **2339** | **78.54%** | **0.1555** | **0.9681** | **4.785 m** | **0.2550** | **2.821 m** | **2.786 m** |
| DistFormer | GT box+LiDAR | all GT | **98.15%** | **0.0567** | **0.08** | **1.260 m** | **0.090** | — | — |

---

## 6. Distance metrics — Class: Cyclist

| ID | Mode | n | δ<1.25↑ | AbsRel↓ | SqRel↓ | RMSE↓ | RMSElog↓ | DE↓ | ALE↓ |
|---|---|---|---|---|---|---|---|---|---|
| V7-val | IoU | 96 | **100.00%** | **0.0256** | **0.0226** | **0.697 m** | **0.0350** | **0.510 m** | **0.599 m** |
| DF-R | IoU+LiDAR | 123 | 96.75% | 0.0907 | 0.1828 | 1.810 m | 0.1061 | 1.399 m | 1.325 m |
| DF-R | Oracle+LiDAR | 681 | 79.74% | 0.1581 | 1.3199 | 6.828 m | 0.2054 | 4.642 m | 4.598 m |
| LID-p1 | IoU+LiDAR | 202 | 98.02% | 0.0753 | 0.1931 | 2.151 m | 0.0927 | 1.554 m | 1.828 m |
| LID-p1 | Oracle+LiDAR | 681 | 79.30% | 0.1571 | 1.5094 | 7.120 m | 0.2244 | 4.625 m | 4.657 m |
| **LID-P2** | **IoU+LiDAR** | **165** | **98.79%** | **0.0843** | **0.1816** | **1.908 m** | **0.0984** | **1.464 m** | **1.541 m** |
| **LID-P2** | **Oracle+LiDAR** | **681** | **80.03%** | **0.1421** | **1.1102** | **6.367 m** | **0.2017** | **4.266 m** | **4.289 m** |
| DistFormer | GT box+LiDAR | all GT | 95.62% | 0.0801 | **0.25** | **3.090 m** | **0.110** | — | — |

---

## 7. Depth (Z) metrics — LID-P2 final (IoU mode)

| Class | n | δ<1.25↑ | AbsRel↓ | SqRel↓ | RMSE↓ | RMSElog↓ |
|---|---|---|---|---|---|---|
| All | 13,085 | 96.19% | 7.50% | 0.340 | 4.007 m | 0.100 |
| Car | 11,912 | 96.26% | 7.40% | 0.356 | 4.166 m | 0.099 |
| Pedestrian | 1,008 | 95.14% | 8.50% | 0.177 | 1.638 m | 0.111 |
| Cyclist | 165 | 96.97% | 8.67% | 0.203 | 1.984 m | 0.105 |

---

## 8. Tóm tắt so sánh OGCDE LID-P2 vs DistFormer

| Class | Mode | OGCDE LID-P2 δ<1.25 | DistFormer δ<1.25 | OGCDE LID-P2 AbsRel | DistFormer AbsRel |
|---|---|---|---|---|---|
| All | IoU | **96.74%** | 93.67% | **7.54%** | 10.39% |
| All | Oracle | 86.58% | **93.67%** | 11.87% | **10.39%** |
| Car | IoU | **96.85%** | 94.32% | **7.44%** | 9.97% |
| Pedestrian | IoU | 95.04% | **98.15%** | 8.50% | **5.67%** |
| Cyclist | IoU | **98.79%** | 95.62% | 8.43% | **8.01%** |

> **IoU mode**: OGCDE thắng All + Car + Cyclist, thua Pedestrian (detection coverage 43%)  
> **Oracle mode**: DistFormer vẫn tốt hơn — distance prediction thuần của DistFormer mạnh hơn OGCDE

---

## 9. Key Observations

1. **OGCDE giải bài toán khó hơn**: Tự detect + predict distance end-to-end, DistFormer chỉ predict distance với GT box cho sẵn.

2. **Oracle mode honest hơn IoU mode**: Oracle = 100% GT objects (cùng coverage DistFormer). IoU = 74-79% objects dễ nhất → số đẹp nhưng có selection bias.

3. **Train GT source quan trọng**: Train LiDAR → δ<1.25 tăng 89.97% → 95.10% (LID-p1) → 96.74% (LID-P2 two-phase).

4. **Two-phase training hiệu quả**: Phase 1 annotation (stability) + Phase 2 LiDAR fine-tune (lr=1e-5, max_grad_norm=2.0) → kết quả tốt nhất, tránh NaN divergence.

5. **V7 tốt nhất tuyệt đối** (annotation GT, split tùy chỉnh): δ<1.25=99.79%, AbsRel=2.19% — nhưng không so sánh fair với DistFormer do khác split và GT source.

6. **Pedestrian khó nhất**: IoU coverage 43% (1008/2339) — object nhỏ, hay bị che, ảnh hưởng metric IoU mode nhiều nhất.

---

## 10. Fairness Summary — OGCDE vs DistFormer

| Yếu tố | DistFormer | OGCDE LID-P2 | Verdict |
|---|---|---|---|
| Train split | 3,712 | 3,711 | ✅ Fair |
| Val split | 3,768 | 3,768 | ✅ Fair |
| Train GT | LiDAR 10th-pct | LiDAR 10th-pct (phase 2) | ✅ Fair |
| Eval GT | LiDAR 10th-pct | LiDAR 10th-pct | ✅ Fair |
| Box input | GT box | Tự detect (IoU) / GT center (Oracle) | ⚠️ Khác nhau |
| Task scope | Distance only | Det + Dist + Depth | ❌ OGCDE khó hơn |
| Model convergence | Fully trained | Converged (~ep430) | ✅ Fair |

---

## 11. Transparency của các Baseline trong DistFormer Paper

> DistFormer lấy số từ paper gốc của các baseline ("Results on KITTI are extracted from their respective papers") mà không enforce cùng điều kiện thí nghiệm.

| Method | Kết quả lấy từ | Split | Box lúc eval | GT distance | Difficulty filter | DontCare filter | Độ tin cậy |
|---|---|---|---|---|---|---|---|
| **SVR** | Paper gốc | Chen et al. ✅ | GT box ✅ | ❓ | ❓ | ❓ | ⚠️ thấp |
| **IPM** | Paper gốc | Chen et al. ✅ | GT box ✅ | ❓ | ❓ | ❓ | ⚠️ thấp |
| **DisNet** | **Tự implement** ✅ | Chen et al. ✅ | GT box ✅ | ❓ | ❓ | ❓ | ⚠️ thấp |
| **Zhu et al.** | Paper gốc | Chen et al. ✅ | GT box ✅ | **LiDAR** ✅ | ❓ | Remove DontCare ✅ | 🟡 trung bình |
| **CenterNet** | Paper gốc | Chen et al. ✅ | GT box ✅ | ❓ | ❓ | ❓ | ⚠️ thấp |
| **PatchNet** | Paper gốc | Chen et al. ✅ | GT box ✅ | ❓ | ❓ | ❓ | ⚠️ thấp |
| **Jing et al.** | Paper gốc | Chen et al. ✅ | GT box ✅ | ❓ | ❓ | ❓ | ⚠️ thấp |
| **DistFormer** | **Tự chạy** ✅ | Chen et al. ✅ | GT box ✅ | **LiDAR 10th-pct** ✅ | ❓ | Remove DontCare ✅ | 🟢 cao |
| **OGCDE LID-P2** | **Tự chạy** ✅ | Chen et al. ✅ | **Tự detect** | **LiDAR 10th-pct** ✅ | Tất cả ✅ | Remove DontCare ✅ | 🟢 cao |

**Chú thích:** ✅ biết chắc | ❓ không rõ | ⚠️ nhiều điểm không rõ | 🟡 biết một phần | 🟢 biết đầy đủ

**Kết luận:** Điểm chắc chắn duy nhất của tất cả baseline là **Chen et al. split** và **GT box làm input**. GT source, difficulty filter, preprocessing chi tiết đều không được ghi rõ vì copy từ paper gốc → Table 1 của DistFormer không phải là apple-to-apple comparison.

---

## 11. Checkpoint & Files

| Model | Checkpoint | Log |
|---|---|---|
| V7 | `runs/ogcde_v7/best.pt` (ep 251, val=8.359) | `runs/ogcde_v7.log` |
| DF-R | `runs/ogcde_distformer/best.pt` (ep ~311, val=18.874) | `runs/ogcde_distformer_resume.log` |
| LID-p1 | `runs/ogcde_distformer_lidar/best.pt` (ep ~280, val=20.565) | `runs/ogcde_distformer_lidar.log` |
| **LID-P2** | **`runs/ogcde_lidar_p2/best.pt`** (ep ~430, val=20.565) | `runs/ogcde_lidar_p2.log` |
| LiDAR GT val | `cache/lidar_gt_val.json` | — |
| LiDAR GT train | `cache/lidar_gt_train.json` | — |
| Split files | `splits/distformer_train.txt`, `splits/distformer_val.txt` | — |
