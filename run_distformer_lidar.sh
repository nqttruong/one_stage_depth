python train.py \
    --kitti-root /media/truong/01DBB45ECE0C4E00/dl/kitti_object/training \
    --train-split /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/distformer_train.txt \
    --val-split /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/distformer_val.txt \
    --dist-source cache/lidar_gt_train.json \
    --backbone-size m \
    --pretrained-backbone yolov8m.pt \
    --freeze-backbone-epochs 10 \
    --epochs 350 --batch 16 --lr 1e-4 --weight-decay 5e-4 \
    --focal-gamma 1.5 --hflip --strong-aug --geo-warmup-epochs 5 \
    --w-box 5.0 --w-obj 2.0 --w-cls 0.5 \
    --cls-weights 1.0 3.0 5.0 \
    --save-dir runs/ogcde_distformer_lidar \
    2>&1 | tee runs/ogcde_distformer_lidar.log
