python train.py \
    --kitti-root /media/truong/01DBB45ECE0C4E00/dl/kitti_object/training \
    --train-split /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/kitti_train.txt \
    --val-split /media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/kitti_val.txt \
    --pretrained-backbone yolov8n.pt \
    --freeze-backbone-epochs 10 \
    --epochs 150 --batch 16 --lr 1e-4 --weight-decay 5e-4 \
    --focal-gamma 1.5 --hflip --geo-warmup-epochs 5 \
    --w-box 5.0 --w-obj 2.0 --w-cls 0.5 \
    --save-dir runs/ogcde_v4
