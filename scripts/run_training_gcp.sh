#!/bin/bash
# GCP VM에서 전체 학습 파이프라인 실행 (Bi-encoder → Cross-encoder)
# 논문 설정 그대로: Bi 20 epochs / batch 16 / n_neg 25, Cross 5 epochs / batch 32 / n_neg 50
# VM 내부에서: cd ~/kb-adaptive && ./scripts/run_training_gcp.sh

set -e
# 스크립트 위치 기준으로 프로젝트 루트로 이동 (어디서 실행해도 동일 동작)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_ROOT="${DATA_ROOT:-dataset_ketqa}"
OUT_BI="${OUT_BI:-outputs/bi_encoder}"
OUT_CROSS="${OUT_CROSS:-outputs/cross_encoder}"
DEVICE="${DEVICE:-cuda}"

if [ ! -d "$DATA_ROOT/data" ]; then
  echo "Dataset not found at $DATA_ROOT. Copy dataset_ketqa to this directory first."
  exit 1
fi

echo "=== Bi-encoder (논문: 20 epochs, batch 16, n_negatives 25) ==="
python train_bi_encoder.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_BI" \
  --device "$DEVICE"

echo "=== Cross-encoder (논문: 5 epochs, batch 32, n_negatives 50) ==="
python train_cross_encoder.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_CROSS" \
  --device "$DEVICE"

echo "=== Training done. Checkpoints: $OUT_BI, $OUT_CROSS ==="
