#!/bin/bash
# Bi-encoder만 학습 (VM 2대 쓸 때 이 VM 전용). gradient checkpointing + fp16으로 논문 배치.
# 사용: cd ~/kb-adaptive && ./scripts/run_bi_only_gcp.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_ROOT="${DATA_ROOT:-dataset_ketqa}"
OUT_BI="${OUT_BI:-outputs/bi_encoder}"
DEVICE="${DEVICE:-cuda}"
BATCH_BI="${BATCH_BI:-16}"
ACCUM_BI="${ACCUM_BI:-1}"
N_NEG_BI="${N_NEG_BI:-25}"

if [ ! -d "$DATA_ROOT/data" ]; then
  echo "Dataset not found at $DATA_ROOT."
  exit 1
fi

echo "=== Bi-encoder only (batch $BATCH_BI, n_neg $N_NEG_BI, fp16) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_bi_encoder.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_BI" \
  --device "$DEVICE" \
  --batch_size "$BATCH_BI" \
  --n_negatives "$N_NEG_BI" \
  --accumulation_steps "$ACCUM_BI" \
  --fp16
echo "=== Bi-encoder done: $OUT_BI ==="
