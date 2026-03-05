#!/bin/bash
# GCP VM에서 전체 학습 파이프라인 실행 (Bi-encoder → Cross-encoder)
# 기본값: L4 24GB에서 OOM 없이 돌리기용 (논문 수치 아님. 논문은 batch 16/32, n_neg 25/50)
# VM 내부에서: cd ~/kb-adaptive && ./scripts/run_training_gcp.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_ROOT="${DATA_ROOT:-dataset_ketqa}"
OUT_BI="${OUT_BI:-outputs/bi_encoder}"
OUT_CROSS="${OUT_CROSS:-outputs/cross_encoder}"
DEVICE="${DEVICE:-cuda}"
# 일단 돌리기용: 메모리 여유 있게 (L4 24GB)
BATCH_BI="${BATCH_BI:-4}"
ACCUM_BI="${ACCUM_BI:-2}"
N_NEG_BI="${N_NEG_BI:-12}"
BATCH_CROSS="${BATCH_CROSS:-4}"
ACCUM_CROSS="${ACCUM_CROSS:-4}"
N_NEG_CROSS="${N_NEG_CROSS:-24}"

if [ ! -d "$DATA_ROOT/data" ]; then
  echo "Dataset not found at $DATA_ROOT. Copy dataset_ketqa to this directory first."
  exit 1
fi

echo "=== Bi-encoder (effective batch $((BATCH_BI * ACCUM_BI)), n_neg $N_NEG_BI) ==="
python train_bi_encoder.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_BI" \
  --device "$DEVICE" \
  --batch_size "$BATCH_BI" \
  --n_negatives "$N_NEG_BI" \
  --accumulation_steps "$ACCUM_BI"

echo "=== Cross-encoder (effective batch $((BATCH_CROSS * ACCUM_CROSS)), n_neg $N_NEG_CROSS) ==="
python train_cross_encoder.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_CROSS" \
  --device "$DEVICE" \
  --batch_size "$BATCH_CROSS" \
  --n_negatives "$N_NEG_CROSS" \
  --accumulation_steps "$ACCUM_CROSS"

echo "=== Training done. Checkpoints: $OUT_BI, $OUT_CROSS ==="
