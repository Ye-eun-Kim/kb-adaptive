#!/bin/bash
# GCP VM에서 전체 학습 파이프라인 실행 (Bi-encoder → Cross-encoder)
# 논문 설정 (batch 16/32, n_neg 25/50). gradient checkpointing + FP16으로 L4 24GB에서 동작.
# VM 내부에서: cd ~/kb-adaptive && ./scripts/run_training_gcp.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_ROOT="${DATA_ROOT:-dataset_ketqa}"
OUT_BI="${OUT_BI:-outputs/bi_encoder}"
OUT_CROSS="${OUT_CROSS:-outputs/cross_encoder}"
DEVICE="${DEVICE:-cuda}"
# 논문과 동일 (checkpointing + fp16으로 L4에서 수용)
BATCH_BI="${BATCH_BI:-16}"
ACCUM_BI="${ACCUM_BI:-1}"
N_NEG_BI="${N_NEG_BI:-25}"
BATCH_CROSS="${BATCH_CROSS:-32}"
ACCUM_CROSS="${ACCUM_CROSS:-1}"
N_NEG_CROSS="${N_NEG_CROSS:-50}"

if [ ! -d "$DATA_ROOT/data" ]; then
  echo "Dataset not found at $DATA_ROOT. Copy dataset_ketqa to this directory first."
  exit 1
fi

echo "=== Bi-encoder (논문: batch $BATCH_BI, n_neg $N_NEG_BI, fp16) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_bi_encoder.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_BI" \
  --device "$DEVICE" \
  --batch_size "$BATCH_BI" \
  --n_negatives "$N_NEG_BI" \
  --accumulation_steps "$ACCUM_BI" \
  --fp16

echo "=== Cross-encoder (논문: batch $BATCH_CROSS, n_neg $N_NEG_CROSS, fp16) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_cross_encoder.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_CROSS" \
  --device "$DEVICE" \
  --batch_size "$BATCH_CROSS" \
  --n_negatives "$N_NEG_CROSS" \
  --accumulation_steps "$ACCUM_CROSS" \
  --fp16

echo "=== Training done. Checkpoints: $OUT_BI, $OUT_CROSS ==="
