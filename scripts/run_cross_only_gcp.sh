#!/bin/bash
# Cross-encoder만 학습 (VM 2대 쓸 때 이 VM 전용). gradient checkpointing + fp16으로 논문 배치.
# 사용: cd ~/kb-adaptive && ./scripts/run_cross_only_gcp.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_ROOT="${DATA_ROOT:-dataset_ketqa}"
OUT_CROSS="${OUT_CROSS:-outputs/cross_encoder}"
DEVICE="${DEVICE:-cuda}"
BATCH_CROSS="${BATCH_CROSS:-32}"
ACCUM_CROSS="${ACCUM_CROSS:-1}"
N_NEG_CROSS="${N_NEG_CROSS:-50}"

if [ ! -d "$DATA_ROOT/data" ]; then
  echo "Dataset not found at $DATA_ROOT."
  exit 1
fi

echo "=== Cross-encoder only (batch $BATCH_CROSS, n_neg $N_NEG_CROSS, fp16) ==="
RESUME_ARGS=""
if [ -n "${RESUME_FROM:-}" ] && [ -f "$RESUME_FROM" ]; then
  RESUME_ARGS="--resume_from $RESUME_FROM"
  echo "Resuming from: $RESUME_FROM"
fi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_cross_encoder.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_CROSS" \
  --device "$DEVICE" \
  --batch_size "$BATCH_CROSS" \
  --n_negatives "$N_NEG_CROSS" \
  --accumulation_steps "$ACCUM_CROSS" \
  --fp16 \
  $RESUME_ARGS
echo "=== Cross-encoder done: $OUT_CROSS ==="
