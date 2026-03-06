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
gcloud compute instances create cross-encoder-train \
  --zone=us-central1-b \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True

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
  --boot-disk-size=200GB \
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
chmod +x scripts/*.sh
./scripts/run_training_gcp.sh
```

백그라운드 실행 (SSH 끊어도 계속):

```bash
nohup ./scripts/run_training_gcp.sh > train.log 2>&1 &
tail -f train.log
```

### 5.5 (선택) VM 2대로 Bi / Cross 나눠서 학습

VM 한 대에서 Bi → Cross 순서로 돌리면 OOM 때문에 배치를 줄여야 합니다. **VM 두 대**를 쓰면 각각 GPU를 하나만 쓰므로 배치를 키울 수 있고, Bi와 Cross를 **동시에** 돌려서 총 시간도 줄일 수 있습니다.

1. **VM 2대 생성** (이름만 다르게, 예: `ketqa-bi`, `ketqa-cross`). 각각 동일하게 클론·환경 세팅·데이터 복사.
2. **VM1 (ketqa-bi)**
  `./scripts/run_bi_only_gcp.sh`  
   (필요하면 `BATCH_BI=8 ACCUM_BI=2 N_NEG_BI=25` 등으로 논문에 가깝게)
3. **VM2 (ketqa-cross)**
  `./scripts/run_cross_only_gcp.sh`  
   (필요하면 `BATCH_CROSS=8 ACCUM_CROSS=4 N_NEG_CROSS=50` 등)
4. 학습 끝난 뒤 각 VM에서 `outputs/bi_encoder`, `outputs/cross_encoder`를 로컬로 받아 한 폴더에 합치면 됩니다.

비용은 2대 동시 사용분이지만, 병렬로 돌리면 총 걸리는 시간이 줄어서 전체 비용은 비슷할 수 있습니다.

---

## 6. 결과 다운로드 (VM → 로컬)

학습이 끝난 뒤 VM에서 로컬로 받을 때 두 가지 방법이 있다.

### 6.1 scp (간단하지만 끊기면 처음부터)

```bash
# 로컬에서 (ZONE, VM 이름 본인 걸로)
gcloud compute scp --recurse ketqa-train:~/kb-adaptive/outputs ./outputs_from_gcp --zone=us-central1-b
# 또는 bi/cross만
gcloud compute scp --recurse ketqa-train:~/kb-adaptive/outputs/bi_encoder ./outputs_bi_from_gcp --zone=us-central1-b
```

**주의:** `scp`는 연결이 끊기면 **이어받기 불가**. 인터넷 끊김·핫스팟 전환 시 전송이 중단되면 **다시 같은 명령으로 처음부터** 받아야 한다. 대용량이거나 연결이 불안정하면 아래 GCS 경유를 쓰는 것이 안전하다.

### 6.2 GCS 경유 (끊겨도 재개 가능, 대용량 권장)

VM → GCS → 로컬 순으로 하면 `gsutil`이 대용량 전송 시 재개를 지원한다.

```bash
# 1) VM 안에서: outputs를 버킷으로 업로드
gsutil -m cp -r ~/kb-adaptive/outputs gs://YOUR_BUCKET/outputs

# 2) 로컬에서: 버킷에서 다운로드 (끊겨도 다시 실행하면 이어받기 가능)
gsutil -m cp -r gs://YOUR_BUCKET/outputs ./outputs_from_gcp
```

버킷에 VM 서비스 계정 접근 권한이 있어야 한다 (이미 데이터 업로드할 때 설정했다면 동일 버킷 사용 가능).

---

## 7. 비용 & VM 중지

- **VM 중지**: Compute Engine → 인스턴스 선택 → **중지**
- **VM 삭제**: 사용 끝나면 인스턴스 삭제해 크레딧 소모 중단
- T4 + n1-standard-8 기준 대략 **$1~2/시간** 수준

---

## 8. 트러블슈팅


| 현상                                               | 조치                                                                                            |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `acceleratorTypes/nvidia-tesla-t4 was not found` | `us-central1-a` 또는 `asia-northeast1-a` 등 T4 지원 zone 사용                                        |
| `nvidia-smi` 없음                                  | VM 부팅 후 2–3분 대기 (드라이버 자동 설치)                                                                  |
| `Dataset not found`                              | `ls dataset_ketqa/data` 로 train.json 있는지 확인, 경로가 `~/kb-adaptive/dataset_ketqa` 인지 확인          |
| CUDA OOM                                         | 기본이 이미 보수적(일단 돌리기용). 그래도 OOM이면 `BATCH_BI=2 N_NEG_BI=8 BATCH_CROSS=2 N_NEG_CROSS=16` 등으로 더 줄이기 |


