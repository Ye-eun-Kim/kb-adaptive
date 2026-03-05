# GCP $300 크레딧으로 KET-QA Retriever 학습 세팅

90일 무료 체험 + $300 크레딧으로 GPU VM을 만들어 학습하는 방법입니다.

**흐름**: 코드는 **GitHub 클론**, 데이터셋은 **.zip을 GCS에 올린 뒤 VM에서 다운로드 & 압축 해제**.

**논문 대조 체크리스트**: [PAPER_CHECKLIST.md](PAPER_CHECKLIST.md)

---

## 빠른 요약 (이미 gcloud 설치·로그인 된 경우)

```bash
export PROJECT_ID=your-gcp-project-id
export ZONE=us-central1-a   # T4 사용 가능 zone (서울 미지원)
gcloud config set project $PROJECT_ID

# 1) GPU VM 생성 (T4, Ubuntu 22.04)
gcloud compute instances create ketqa-train --zone=$ZONE \
  --machine-type=n1-standard-8 --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB --metadata=install-nvidia-driver=True \
  --maintenance-policy=TERMINATE

# 2) GCS 버킷 생성 (한 번만) + 데이터셋 zip 업로드 (로컬에서)
gsutil mb -l us-central1 gs://YOUR_BUCKET
gsutil cp dataset_ketqa.zip gs://YOUR_BUCKET/

# 3) VM 접속 후: 클론 → 세팅 → 데이터 받기 → 학습
gcloud compute ssh ketqa-train --zone=$ZONE
# VM 안에서 아래 "5. VM 안에서 할 일" 순서대로 실행
```

---

## 1. 사전 준비 (로컬)

- [Google Cloud SDK (gcloud)](https://cloud.google.com/sdk/docs/install) 설치 후 `gcloud init` 로 로그인
- 코드는 **GitHub에 푸시**된 상태, 데이터셋은 **dataset_ketqa.zip** 으로 압축해 둔 상태

---

## 2. 프로젝트 & 결제

1. [Cloud Console](https://console.cloud.google.com/) → 상단 프로젝트 선택/생성
2. **결제** 연결 (무료 체험 $300 자동 적용)
3. **API 사용 설정**
  - **Compute Engine API** 사용
  - **Cloud Storage API** 사용 (GCS로 데이터셋 전달 시 필요)

---

## 3. GPU VM 생성

```bash
export PROJECT_ID=your-project-id
export ZONE=us-central1-a
gcloud config set project $PROJECT_ID

gcloud compute instances create ketqa-train \
  --zone=$ZONE \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-standard \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True
```

- **T4 사용 가능 zone**: `us-central1-a`, `asia-northeast1-a`(도쿄) 등. **서울(asia-northeast3)에는 T4 없음.**

---

## 4. 데이터셋을 GCS로 올리기 (로컬에서)

데이터셋은 Git에 없고 **.zip**으로 GCS에 올린 뒤, VM에서 받아 씁니다.

```bash
# 버킷 한 번만 생성 (이미 있으면 생략)
gsutil mb -l us-central1 gs://YOUR_BUCKET

# dataset_ketqa.zip 업로드 (프로젝트 루트에 zip 있다고 가정)
gsutil cp dataset_ketqa.zip gs://YOUR_BUCKET/
```

---

## 5. VM 안에서 할 일 (순서대로)

VM 생성 후 **2–3분** 기다린 뒤 SSH 접속합니다.

```bash
gcloud compute ssh ketqa-train --zone=$ZONE
```

### 5.1 GitHub 클론

```bash
git clone https://github.com/YOUR_USER/kb-adaptive.git
cd kb-adaptive
```

(`YOUR_USER` / `kb-adaptive` 는 실제 저장소 URL로 바꿉니다.)

### 5.2 환경 세팅 (한 번만)

```bash
chmod +x scripts/*.sh
./scripts/gcp_setup_vm.sh
```

Python 3.11, venv, PyTorch(CUDA), transformers 등이 설치됩니다.

### 5.3 GCS에서 데이터셋 받기

```bash
# YOUR_BUCKET, YOUR_ZONE 은 본인 값으로
gsutil cp gs://YOUR_BUCKET/dataset_ketqa.zip .
unzip -q dataset_ketqa.zip
# 압축 해제 결과가 dataset_ketqa/ 폴더가 되도록 (zip 루트에 data/, tables/, entity_base/ 있으면 됨)
ls dataset_ketqa/data
ls dataset_ketqa/tables
ls dataset_ketqa/entity_base
```

압축 구조가 `dataset_ketqa/data`, `dataset_ketqa/tables` 등이 아니라 **zip 루트가 data/tables/entity_base** 인 경우:

```bash
unzip -q dataset_ketqa.zip
mv data tables entity_base dataset_ketqa 2>/dev/null || true
# 또는 zip 내용에 맞게 dataset_ketqa/ 한 폴더로 정리
```

최종적으로 `~/kb-adaptive/dataset_ketqa/data/train.json`, `dataset_ketqa/tables/`, `dataset_ketqa/entity_base/` 가 있으면 됩니다.

### 5.4 학습 실행 (논문 설정 그대로)

```bash
source ~/venv/bin/activate
./scripts/run_training_gcp.sh
```

백그라운드 실행 (SSH 끊어도 계속):

```bash
nohup ./scripts/run_training_gcp.sh > train.log 2>&1 &
tail -f train.log
```

---

## 6. 결과 다운로드 (VM → 로컬)

학습이 끝난 뒤:

```bash
# 로컬에서
gcloud compute scp --recurse ketqa-train:~/kb-adaptive/outputs ./outputs_from_gcp --zone=$ZONE
```

---

## 7. 비용 & VM 중지

- **VM 중지**: Compute Engine → 인스턴스 선택 → **중지**
- **VM 삭제**: 사용 끝나면 인스턴스 삭제해 크레딧 소모 중단
- T4 + n1-standard-8 기준 대략 **$1~2/시간** 수준

---

## 8. 트러블슈팅


| 현상                                               | 조치                                                                                                       |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `acceleratorTypes/nvidia-tesla-t4 was not found` | `us-central1-a` 또는 `asia-northeast1-a` 등 T4 지원 zone 사용                                                   |
| `nvidia-smi` 없음                                  | VM 부팅 후 2–3분 대기 (드라이버 자동 설치)                                                                             |
| `Dataset not found`                              | `ls dataset_ketqa/data` 로 train.json 있는지 확인, 경로가 `~/kb-adaptive/dataset_ketqa` 인지 확인                     |
| CUDA OOM (L4 24GB 등)                             | 논문 effective batch 유지: `BATCH_BI=8 ACCUM_BI=2 BATCH_CROSS=8 ACCUM_CROSS=4 ./scripts/run_training_gcp.sh` |


