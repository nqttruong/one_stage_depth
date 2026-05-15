#!/usr/bin/env bash
# Ablation chain: retrain V2 → eval → V3 → eval → V4 → eval
# Pre-flight clean each variant dir before training.
# Stops on any failure.
set -euo pipefail

KITTI_ROOT=/media/truong/01DBB45ECE0C4E00/dl/kitti_object/training
SPLIT=/media/truong/01DBB45ECE0C4E00/dl/kitti_object/splits/distformer_val.txt
LIDAR_GT=cache/lidar_gt_val.json
LOG_DIR=logs
RESULT_DIR=results
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# V1 AbsRel (already trained + verified)
V1_ABSREL=0.1309

# ─── helpers ──────────────────────────────────────────────────────────────────

die() { echo "❌ FATAL: $*" >&2; exit 1; }

# Show val_loss at multiples of 10 epochs from training log (proxy for monitoring)
show_val_curve() {
    local log="$1"
    echo "  Val-loss every 10 epochs:"
    grep "val total" "$log" | awk 'NR%10==0 {printf "    %s\n", $0}'
}

# Run full eval and return AbsRel for a checkpoint
eval_ckpt() {
    local ckpt="$1"
    PYTHONPATH=. python scripts/eval_ablation.py \
        --kitti-root "$KITTI_ROOT" \
        --split "$SPLIT" \
        --lidar-gt "$LIDAR_GT" \
        --out-dir "$RESULT_DIR" 2>/dev/null
    # Extract just the variant's line
    grep "^v" "$RESULT_DIR/ablation.csv" | grep "$(basename "$(dirname "$ckpt")")"
}

train_variant() {
    local variant="$1"
    local config="configs/ablation/${variant}.yaml"
    local run_dir="runs/ablation/${variant}"
    local log="${LOG_DIR}/${variant}.log"
    local start_ts

    [ -f "$config" ] || die "Config not found: $config"

    # Pre-flight clean
    if [ -d "$run_dir" ]; then
        echo "  Cleaning stale directory: $run_dir"
        rm -rf "$run_dir"
    fi
    mkdir -p "$run_dir"

    start_ts=$(date +%s)
    echo "=== ${variant} RETRAIN starting $(date) ===" | tee "$log"
    PYTHONPATH=. python train.py --config "$config" 2>&1 | tee -a "$log"
    echo "=== ${variant} RETRAIN finished $(date) ===" | tee -a "$log"

    # Verify checkpoint is newer than start time
    local ckpt="${run_dir}/best.pt"
    [ -f "$ckpt" ] || die "No best.pt found after training $variant"
    local ckpt_ts
    ckpt_ts=$(stat -c %Y "$ckpt")
    if [ "$ckpt_ts" -lt "$start_ts" ]; then
        die "best.pt timestamp ($ckpt_ts) < start time ($start_ts) — stale checkpoint!"
    fi
    echo "  ✅ Checkpoint verified: $(date -d @"$ckpt_ts") > start $(date -d @"$start_ts")"
}

eval_variant() {
    local variant="$1"
    local ckpt="runs/ablation/${variant}/best.pt"
    echo ""
    echo "--- Eval: $variant ---"
    PYTHONPATH=. python - << PYEOF
import json, torch, numpy as np
from ogcde.model import OGCDENet
from ogcde.dataset import KITTIOGCDEDataset, collate_ogcde
from ogcde.utils import decode_predictions, unletterbox_boxes
from ogcde.metrics import xywh_to_xyxy, bbox_iou_xyxy
from torch.utils.data import DataLoader

KITTI   = '$KITTI_ROOT'
SPLIT   = '$SPLIT'
LIDAR   = json.load(open('$LIDAR_GT'))
DEVICE  = 'cuda'
IOU_THR = 0.5
CONF    = 0.25

ck = torch.load('$ckpt', map_location=DEVICE, weights_only=False)
use_depth = ck.get('args', {}).get('no_s_head', False)
nc = ck.get('args', {}).get('num_classes', 3)
bs = ck.get('args', {}).get('backbone_size', 'n')
model = OGCDENet(nc=nc, backbone_size=bs).cuda()
model.load_state_dict(ck['model']); model.eval()

ds = KITTIOGCDEDataset(KITTI, SPLIT, img_size=640, augment=False)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_ogcde)

preds, gts = [], []
with torch.no_grad():
    for imgs, targets, meta in loader:
        imgs = imgs.cuda()
        dets = decode_predictions(model(imgs), nc, obj_thr=CONF, iou_thr=IOU_THR)
        for b, det in enumerate(dets):
            m = meta[b]; bmask = targets['batch_idx'].numpy() == b
            gt_boxes = targets['boxes'].numpy()[bmask]
            gt_labels = targets['labels'].numpy()[bmask]
            gt_obj = targets['obj_idx'].numpy()[bmask]
            pb = unletterbox_boxes(det['boxes'].cpu().numpy(), m['ratio'], m['pad'])
            px = xywh_to_xyxy(pb)
            gx = xywh_to_xyxy(unletterbox_boxes(gt_boxes, m['ratio'], m['pad']))
            pd = (det['depth'] if use_depth else det['distance']).cpu().numpy()
            pc = det['classes'].cpu().numpy()
            if not len(px) or not len(gx): continue
            ious = bbox_iou_xyxy(px, gx)
            lr_db = {r['obj_idx']: r for r in LIDAR.get(str(m['image_id']), [])}
            for pi in range(len(px)):
                row = ious[pi].copy(); row[gt_labels != int(pc[pi])] = -1
                if row.max() < IOU_THR: continue
                gi = int(row.argmax())
                lr = lr_db.get(int(gt_obj[gi]))
                if not lr or lr.get('n_points', 0) == 0: continue
                preds.append(float(pd[pi])); gts.append(float(lr['dist_lidar']))

p, g = np.array(preds), np.array(gts)
absrel = float(np.mean(np.abs(p - g) / g))
delta  = float(np.mean(np.maximum(p/g, g/p) < 1.25))
epoch  = ck.get('epoch', -1)
print(f"RESULT epoch={epoch}  AbsRel={absrel*100:.4f}  delta={delta*100:.4f}  n={len(p)}")
PYEOF
}

compare_with_v1() {
    local variant="$1"
    local absrel_line
    # extract the RESULT line printed by eval_variant
    absrel_line=$(grep "^RESULT" /tmp/_eval_out 2>/dev/null || echo "")
    if [ -z "$absrel_line" ]; then return; fi
    local absrel
    absrel=$(echo "$absrel_line" | grep -oP "AbsRel=\K[0-9.]+")
    local cmp
    cmp=$(python3 -c "print('BETTER' if float('$absrel') < float('$V1_ABSREL')*100 else 'WORSE_OR_EQUAL')")
    echo ""
    echo "  V1 AbsRel = $(python3 -c "print(f'{$V1_ABSREL*100:.4f}')") %"
    echo "  $variant AbsRel = ${absrel} %  → $cmp than V1"
    if [ "$cmp" = "WORSE_OR_EQUAL" ]; then
        echo "  📝 Narrative: CP alone hurts (or neutral). Only useful combined with s-head."
        echo "  Proceeding to V3 with this narrative in mind."
    fi
}

# ─── MAIN ─────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "Ablation chain: V2 retrain → V3 → V4"
echo "V1 AbsRel baseline (verified): $(python3 -c "print(f'{$V1_ABSREL*100:.2f}')") %"
echo "================================================================"

# ── V2 retrain ────────────────────────────────────────────────────────────────
echo ""
echo "▶ Phase 1: Retrain V2 (+ contact point, no sec θ)"
train_variant v2_cp
show_val_curve "$LOG_DIR/v2_cp.log"
eval_variant v2_cp 2>&1 | tee /tmp/_eval_out
compare_with_v1 v2_cp
echo ""
echo "✅ V2 done. Proceeding to V3."

# ── V3 ────────────────────────────────────────────────────────────────────────
echo ""
echo "▶ Phase 2: Train V3 (+ sec θ head)"
train_variant v3_sec_theta
show_val_curve "$LOG_DIR/v3_sec_theta.log"
eval_variant v3_sec_theta 2>&1 | tee /tmp/_eval_out
echo "✅ V3 done. Proceeding to V4."

# ── V4 ────────────────────────────────────────────────────────────────────────
echo ""
echo "▶ Phase 3: Train V4 (+ λ_geo warmup)"
train_variant v4_warmup
show_val_curve "$LOG_DIR/v4_warmup.log"
eval_variant v4_warmup 2>&1 | tee /tmp/_eval_out
echo "✅ V4 done."

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Final ablation eval (all variants)"
echo "================================================================"
PYTHONPATH=. python scripts/eval_ablation.py \
    --kitti-root "$KITTI_ROOT" \
    --split "$SPLIT" \
    --lidar-gt "$LIDAR_GT" \
    --out-dir "$RESULT_DIR"

echo ""
echo "Results saved: $RESULT_DIR/ablation.csv  $RESULT_DIR/ablation_table.tex"
