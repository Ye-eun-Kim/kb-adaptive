# KET-QA Retriever (MKBR)

논문 KET-QA의 **Multistage KB Retriever (MKBR)** 구현입니다.

## 구조

- **Bi-encoder**: 2개의 BERT (question encoder + context encoder), dot product 유사도, contrastive loss  
  - \( s(t, q, T) = E_q(q)^\top E_c(T^* \oplus t^*) \)
- **Cross-encoder**: 1개의 RoBERTa, \( q \oplus T^* \oplus t^* \) 입력, binary classification (logit 점수)  
  - \( s(t, q, T) = E(q \oplus T^* \oplus t^*) \)

## 입력 직렬화

- **Triple**: Relational `[HEAD] ℓ(e1) [REL] ℓ(r) [TAIL] ℓ(e2)`, Attribute `[HEAD] ℓ(e) [REL] ℓ(a) [TAIL] v`
- **Table**: \( T^* = [HEAD], c_1, \ldots, c_N, [ROW], 1, r_1, [ROW], 2, r_2, \ldots \)
- **Sub-table**: triple의 head entity가 등장하는 row만 사용 (row filtering)

## 학습 설정 (논문 기준)

| | Bi-encoder | Cross-encoder |
|---|------------|----------------|
| 모델 | 2× BERT-base-uncased | 1× RoBERTa-base |
| Epochs | 20 | 5 |
| Batch size | 16 | 32 |
| Negative sampling | kNS, n=25 | random, n=50 |
| 공통 | AdamW 1e-5, linear schedule, warm-up, dropout 0.1 |

## 사용법

```bash
# 의존성
pip install -r requirements-retriever.txt

# Bi-encoder 학습
python train_bi_encoder.py --data_root dataset_ketqa --output_dir outputs/bi_encoder

# Cross-encoder 학습
python train_cross_encoder.py --data_root dataset_ketqa --output_dir outputs/cross_encoder

# 소량 실험 (예: 500 = train.json에서 (question, table) 500개만 사용)
python train_bi_encoder.py --data_root dataset_ketqa --max_train_items 500 --epochs 2 --batch_size 4
python train_cross_encoder.py --data_root dataset_ketqa --max_train_items 500 --epochs 1 --batch_size 8

# 평가 (Recall@k)
python eval_retriever.py --data_root dataset_ketqa --split dev

# 단일 질의 추론
python run_retrieval.py --question "What was the release date?" --table_id "Gene_Tierney_1" --top_k 20
```

### 맥북 M4 Pro / 16GB RAM

- `--batch_size`: Bi-encoder **4~8**, Cross-encoder **8~16** 권장 (기본 16/32는 OOM 가능).
- `--device` 미지정 시 `mps`(Mac GPU) 자동 사용. 문제 있으면 `--device cpu`.

### kNS (kNN Negative Sampling)

- 논문: Bi-encoder 학습 시 질문과 **가까운** non-positive triple n=25개를 negative로 사용.
- **현재 구현은 random negative만 동작**합니다. `negative_sampling="knn"` 옵션은 받지만, 실제로는 무시되고 random으로 뽑습니다.
- kNS를 쓰려면: (1) pretrained sentence encoder로 각 (question, neg_context) 임베딩, (2) 질문과 유사도 상위 n개를 negative로 고정해 두고, (3) 데이터셋에서 그 precompute된 negative 리스트를 로드하도록 수정해야 합니다. 지금 코드에서는 **수정 없이 random으로 학습 가능**하며, 논문 수치에 가깝게 하려면 위 kNS 단계를 추가하면 됩니다.

## Adaptive-k (Fixed-k vs 동적 k)

논문 "Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k" 방식을 KET-QA에 적용. **단위는 토큰이 아니라 triple 개수**로 적용.

- **Fixed-k baseline**: k = 5, 10, 20, 50, 100 등 고정 후 Recall 측정.
- **Adaptive-k (ours)**: 정렬된 점수 분포에서 largest gap 위치로 k를 동적 결정 후 Recall 측정.

```bash
python eval_fixed_k_and_adaptive_k.py --data_root dataset_ketqa --split dev --output_dir outputs/eval_adaptive_k
```

결과는 `outputs/eval_adaptive_k/comparison_table.json`, `comparison_table.csv`에 저장됨.

## 디렉터리

- `serialization.py`: Triple/Table 직렬화, sub-table 추출
- `kb.py`: entity_base에서 sub-graph triples 생성
- `dataset.py`: KET-QA retrieval 데이터셋 (positive/negative 샘플)
- `bi_encoder.py`: Bi-encoder 모델 및 contrastive loss
- `cross_encoder.py`: Cross-encoder 모델
- `mkbr.py`: Bi-encoder top-N → Cross-encoder re-rank 파이프라인
- `adaptive_k.py`: Adaptive-k (largest gap, triple 단위 k)
