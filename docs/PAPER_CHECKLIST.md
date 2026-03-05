# KET-QA Retriever vs 논문 최종 대조

GCP에서 논문 설정 그대로 실험하기 전 점검용 체크리스트.

## 1. 모델 구조 (Section 4.4)

| 항목 | 논문 | 구현 | 비고 |
|------|------|------|------|
| Bi-encoder | 2 BERT (base, uncased), [CLS] | `BiEncoder`: query_encoder + context_encoder, last_hidden_state[:,0,:] | ✓ |
| Score (Bi) | s = Eq(q)⊤ Ec(T*⊕t*) | dot product of [CLS] vectors | ✓ |
| Cross-encoder | 1 RoBERTa, q⊕T*⊕t* | `CrossEncoder`: RoBERTa, input = q + context | ✓ |
| Score (Cross) | logit, binary classification | classifier head, logits[:, 1] | ✓ |

## 2. 입력 직렬화 (Section 4.3)

| 항목 | 논문 | 구현 |
|------|------|------|
| Relational triple | [HEAD], ℓ(e1), [REL], ℓ(r), [TAIL], ℓ(e2) | `serialize_relational_triple()` |
| Attribute triple | [HEAD], ℓ(e), [REL], ℓ(a), [TAIL], v | `serialize_attribute_triple()` |
| Table | T* = [HEAD], c1..cN, [ROW], 1, r1, ... | `serialize_table()` |
| Sub-table | head entity가 등장하는 row만 | `table_to_serialized_with_subtable(header, data, head_entity_id)` |

## 3. 학습 설정 (Appendix C.2)

| 항목 | 논문 | config / 학습 스크립트 |
|------|------|------------------------|
| Optimizer | Adam, lr 10⁻⁵ | AdamW, lr=1e-5, weight_decay=0.01 |
| Scheduler | linear, warm-up | get_linear_schedule_with_warmup, warmup_ratio=0.1 |
| Dropout | 0.1 | DROPOUT_RATE=0.1 |
| Bi-encoder | 20 epochs, batch 16, n=25 (kNS) | 20, 16, 25 (negative: random, kNS 미구현) |
| Cross-encoder | 5 epochs, batch 32, n=50 (random) | 5, 32, 50 |

## 4. 추론 (Section 5.1)

| 항목 | 논문 | 구현 |
|------|------|------|
| Bi-encoder로 뽑는 후보 수 | N=200 | top_n_bi=200, N_RETRIEVED_TRIPLES=200 |
| Re-rank 후 반환 | top-k | top_k_final (기본 20) |

## 5. GCP 실행 시 확인

- `scripts/run_training_gcp.sh`: `--max_train_items` 없음 → **전체 train** 사용.
- `--batch_size`, `--n_negatives` 미지정 → **config 기본값(논문과 동일)** 사용.
- 스크립트 실행 전 `cd` 로 프로젝트 루트 이동 → **어디서 실행해도 동작**.

## 6. 논문과의 차이 (허용)

- **Negative sampling**: 논문은 Bi-encoder에 kNS(n=25), Cross는 random(n=50). 구현은 Bi/Cross 모두 **random** (kNS 미구현). 성능 차이는 있을 수 있으나 실험 가능.
- **Optimizer**: 논문 "Adam" → 구현 **AdamW + weight_decay 0.01** (BERT 파인튜닝에서 일반적).
