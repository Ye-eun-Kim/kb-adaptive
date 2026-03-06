# 실험 설정 정리 (GCP에서 실제로 돌린 설정)

결과 해석 시 "어떤 config로 돌렸는지" 참고용.  
**정확한 값은 각 실험의 `outputs/*/run_config.json`으로 확인** (해당 실험 이후에 run_config 저장이 추가됨).

---

## Bi-encoder (실제로 돌린 설정)

- **스크립트**: `run_training_gcp.sh` (전체 파이프라인) 당시 버전
- **당시 스크립트 상태** (이미지에 나온 "일단 돌리기용" 주석 버전):
  - `BATCH_BI=4`, `ACCUM_BI=2` → **effective batch 8**
  - `N_NEG_BI=12`
- **다른 가능성**: 그 후 OOM으로 스크립트를 더 줄인 버전으로 돌았다면  
`BATCH_BI=2`, `ACCUM_BI=4`, `N_NEG_BI=8` (effective 8) 일 수 있음.

**정리**: Bi는 **effective 8, n_neg 12** (이미지 기준) 또는 **effective 8, n_neg 8** (2/4/8 버전) 중 하나.  
→ `outputs/bi_encoder/run_config.json` 이 있으면 그걸로 확정.


| 항목                 | 이미지(예전) 버전 | 현재 run_training_gcp.sh |
| ------------------ | ---------- | ---------------------- |
| batch_size         | 4          | 2                      |
| accumulation_steps | 2          | 4                      |
| effective_batch    | 8          | 8                      |
| n_negatives        | 12         | 8                      |


---

## Cross-encoder (실제로 돌린 설정)

- **스크립트**: `**run_cross_only_gcp.sh`** (Cross 전용, 최종 수정본)
- **설정**:
  - `BATCH_CROSS=1`
  - `ACCUM_CROSS=16` → **effective batch 16**
  - `N_NEG_CROSS=8`


| 항목                 | 값   |
| ------------------ | --- |
| batch_size         | 1   |
| accumulation_steps | 16  |
| effective_batch    | 16  |
| n_negatives        | 8   |


(논문: batch 32, n_neg 50 → L4 24GB OOM 때문에 위처럼 축소한 설정.)

---

## 확인 방법

- **이미 실험한 run**: `outputs/bi_encoder/run_config.json`, `outputs/cross_encoder/run_config.json` 에 실제 사용된 인자와 `started_at` 저장됨 (해당 기능 추가 이후 실행분만).
- **앞으로 실험**: 실행 시마다 위 경로에 config 자동 저장됨.

