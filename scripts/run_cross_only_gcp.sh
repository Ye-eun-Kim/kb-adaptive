#!/bin/bash
# Cross-encoder만 학습 (VM 2대 쓸 때 이 VM 전용)
# 사용: cd ~/kb-adaptive && ./scripts/run_cross_only_gcp.sh
# L4 24GB 기본값: batch 1, n_neg 8 (OOM 방지). A100이면 BATCH_CROSS=8 ACCUM_CROSS=4 N_NEG_CROSS=50

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_ROOT="${DATA_ROOT:-dataset_ketqa}"
OUT_CROSS="${OUT_CROSS:-outputs/cross_encoder}"
DEVICE="${DEVICE:-cuda}"
BATCH_CROSS="${BATCH_CROSS:-1}"
ACCUM_CROSS="${ACCUM_CROSS:-16}"
N_NEG_CROSS="${N_NEG_CROSS:-8}"

if [ ! -d "$DATA_ROOT/data" ]; then
  echo "Dataset not found at $DATA_ROOT."
  exit 1
fi

echo "=== Cross-encoder only (effective batch $((BATCH_CROSS * ACCUM_CROSS)), n_neg $N_NEG_CROSS) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_cross_encoder.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_CROSS" \
  --device "$DEVICE" \
  --batch_size "$BATCH_CROSS" \
  --n_negatives "$N_NEG_CROSS" \
  --accumulation_steps "$ACCUM_CROSS"
echo "=== Cross-encoder done: $OUT_CROSS ==="
