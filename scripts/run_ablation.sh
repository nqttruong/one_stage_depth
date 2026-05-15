#!/usr/bin/env bash
# Run 4 ablation variants sequentially (single GPU).
# Resumable: skips variant if best.pt already exists.
set -euo pipefail

VARIANTS=(v1_baseline v2_cp v3_sec_theta v4_warmup)
mkdir -p logs

for V in "${VARIANTS[@]}"; do
    CKPT="runs/ablation/${V}/best.pt"
    if [ -f "$CKPT" ]; then
        echo "=== ${V}: already complete (${CKPT} exists), skipping ==="
        continue
    fi
    echo "=== ${V} starting $(date) ==="
    python train.py \
        --config "configs/ablation/${V}.yaml" \
        2>&1 | tee "logs/${V}.log"
    echo "=== ${V} finished $(date) ==="
done

echo "All variants done. Run scripts/eval_ablation.py to collect results."
