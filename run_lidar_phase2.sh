# Phase 2: Fine-tune annotation-pretrained model on LiDAR GT
# Starting from runs/ogcde_distformer/best.pt (annotation GT, converged ep~311)
# Low lr + tight grad clip to handle LiDAR GT noise

python train.py \
    --resume runs/ogcde_distformer/best.pt \
    --reset-best \
    --backbone-size m \
    --kitti-root /media/truong/01DBB45ECE0C4E00/dl/kitti_object/training \
    --train-split /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/distformer_train.txt \
    --val-split /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/distformer_val.txt \
    --dist-source cache/lidar_gt_train.json \
    --epochs 450 --batch 16 --lr 1e-5 --weight-decay 5e-4 \
    --focal-gamma 1.5 --hflip --strong-aug --geo-warmup-epochs 0 \
    --w-box 5.0 --w-obj 2.0 --w-cls 0.5 \
    --cls-weights 1.0 3.0 5.0 \
    --freeze-backbone-epochs 0 \
    --max-grad-norm 2.0 \
    --save-dir runs/ogcde_lidar_p2 \
    2>&1 | tee runs/ogcde_lidar_p2.log
