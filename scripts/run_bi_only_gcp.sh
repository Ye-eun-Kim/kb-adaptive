#!/bin/bash
# Bi-encoder만 학습 (VM 2대 쓸 때 이 VM 전용. GPU 하나만 쓰므로 배치 키워도 됨)
# 사용: cd ~/kb-adaptive && ./scripts/run_bi_only_gcp.sh
# 논문에 가깝게: BATCH_BI=16 ACCUM_BI=1 N_NEG_BI=25 (L4 24GB면 8/2/25 권장)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_ROOT="${DATA_ROOT:-dataset_ketqa}"
OUT_BI="${OUT_BI:-outputs/bi_encoder}"
DEVICE="${DEVICE:-cuda}"
BATCH_BI="${BATCH_BI:-8}"
ACCUM_BI="${ACCUM_BI:-2}"
N_NEG_BI="${N_NEG_BI:-25}"

if [ ! -d "$DATA_ROOT/data" ]; then
  echo "Dataset not found at $DATA_ROOT."
  exit 1
fi

echo "=== Bi-encoder only (effective batch $((BATCH_BI * ACCUM_BI)), n_neg $N_NEG_BI) ==="
python train_bi_encoder.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_BI" \
  --device "$DEVICE" \
  --batch_size "$BATCH_BI" \
  --n_negatives "$N_NEG_BI" \
  --accumulation_steps "$ACCUM_BI"
echo "=== Bi-encoder done: $OUT_BI ==="
