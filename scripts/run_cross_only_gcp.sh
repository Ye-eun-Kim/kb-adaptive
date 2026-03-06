#!/bin/bash
# Cross-encoder만 학습 (VM 2대 쓸 때 이 VM 전용. GPU 하나만 쓰므로 배치 키워도 됨)
# 사용: cd ~/kb-adaptive && ./scripts/run_cross_only_gcp.sh
# 논문에 가깝게: BATCH_CROSS=32 ACCUM_CROSS=1 N_NEG_CROSS=50 (L4 24GB면 8/4/50 권장)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_ROOT="${DATA_ROOT:-dataset_ketqa}"
OUT_CROSS="${OUT_CROSS:-outputs/cross_encoder}"
DEVICE="${DEVICE:-cuda}"
BATCH_CROSS="${BATCH_CROSS:-8}"
ACCUM_CROSS="${ACCUM_CROSS:-4}"
N_NEG_CROSS="${N_NEG_CROSS:-50}"

if [ ! -d "$DATA_ROOT/data" ]; then
  echo "Dataset not found at $DATA_ROOT."
  exit 1
fi

echo "=== Cross-encoder only (effective batch $((BATCH_CROSS * ACCUM_CROSS)), n_neg $N_NEG_CROSS) ==="
python train_cross_encoder.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_CROSS" \
  --device "$DEVICE" \
  --batch_size "$BATCH_CROSS" \
  --n_negatives "$N_NEG_CROSS" \
  --accumulation_steps "$ACCUM_CROSS"
echo "=== Cross-encoder done: $OUT_CROSS ==="
