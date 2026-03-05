#!/bin/bash
# GCP VM (Ubuntu 22.04 + NVIDIA T4) 에서 한 번만 실행하는 초기 세팅
# 사용법: chmod +x gcp_setup_vm.sh && ./gcp_setup_vm.sh

set -e

echo "[1/5] System update..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl

echo "[2/5] NVIDIA driver check..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "NVIDIA driver not found. Wait a few minutes after VM boot or add metadata install-nvidia-driver=True and reboot."
  exit 1
fi
nvidia-smi

echo "[3/5] Python 3.10 + venv..."
sudo apt-get install -y -qq python3.10 python3.10-venv python3-pip
PYTHON=$(which python3.10 || which python3)
$PYTHON -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip

echo "[4/5] PyTorch (CUDA 11.8)..."
# T4는 CUDA 11.x / 12.x 호환
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers tqdm

echo "[5/5] Project & dependencies..."
# 스크립트가 scripts/gcp_setup_vm.sh 로 있을 때 프로젝트 루트 = 상위 디렉터리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$PROJECT_ROOT/requirements-retriever.txt" ]; then
  cd "$PROJECT_ROOT"
  pip install -r requirements-retriever.txt
  echo "Installed deps from $PROJECT_ROOT"
else
  echo "Project root not found ($PROJECT_ROOT). Copy kb-adaptive repo to VM first."
fi

echo ""
echo "Done. Next steps:"
echo "  1) Copy dataset_ketqa to VM: gcloud compute scp --recurse dataset_ketqa ketqa-train:~/kb-adaptive/"
echo "  2) SSH: gcloud compute ssh ketqa-train --zone=YOUR_ZONE"
echo "  3) Run: source ~/venv/bin/activate && cd kb-adaptive && ./scripts/run_training_gcp.sh"
