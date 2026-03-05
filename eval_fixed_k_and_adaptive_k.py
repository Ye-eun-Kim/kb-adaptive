#!/usr/bin/env python3
"""
Fixed-k baseline vs Adaptive-k (ours) Recall 비교.
- Fixed-k: k = 5, 10, 20, 50, 100 등 지정하여 top-k triple만 사용, Recall 측정.
- Adaptive-k: 논문 방식으로 점수 분포의 largest gap으로 k를 동적 결정 후 Recall 측정.
- 결과 표를 CSV/JSON으로 저장.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever.config import set_data_paths
from retriever.dataset import load_qa_split
from retriever.kb import evidence_to_triple_dict
from retriever.mkbr import MKBR
from retriever.bi_encoder import BiEncoder
from retriever.cross_encoder import CrossEncoder
from retriever.adaptive_k import adaptive_k_largest_gap
from transformers import BertTokenizer, RobertaTokenizer


def get_gold_serialized(item: dict) -> set:
    gold = set()
    for ev in item.get("selected_evidence_candidate", []):
        td = evidence_to_triple_dict(ev)
        if td:
            gold.add(td["serialized"])
    return gold


def recall_at_k(retrieved_serialized: list, gold: set, k: int) -> float:
    if not gold:
        return 1.0
    top_k = retrieved_serialized[:k]
    hit = len(gold & set(top_k))
    return hit / len(gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset_ketqa")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--bi_encoder_path", type=str, default="outputs/bi_encoder")
    parser.add_argument("--cross_encoder_path", type=str, default="outputs/cross_encoder")
    parser.add_argument("--bi_ckpt", type=str, default="epoch_20.pt")
    parser.add_argument("--cross_ckpt", type=str, default="epoch_5.pt")
    parser.add_argument("--top_n", type=int, default=200, help="Bi-encoder 후 cross-encoder re-rank할 후보 수")
    parser.add_argument("--fixed_k_list", type=str, default="5,10,20,50,100", help="Fixed-k baseline에 쓸 k 목록 (쉼표)")
    parser.add_argument("--adaptive_top_frac", type=float, default=0.9, help="Adaptive-k: largest gap 검사 구간 (상위 비율)")
    parser.add_argument("--adaptive_buffer", type=int, default=5, help="Adaptive-k: k에 더할 buffer")
    parser.add_argument("--max_eval", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_adaptive_k")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    fixed_k_list = [int(x.strip()) for x in args.fixed_k_list.split(",")]

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        )

    set_data_paths(args.data_root)
    from retriever.config import TABLE_DIR, ENTITY_BASE_DIR, QA_DATA_DIR

    tokenizer_q = BertTokenizer.from_pretrained(args.bi_encoder_path)
    tokenizer_ctx = BertTokenizer.from_pretrained(os.path.join(args.bi_encoder_path, "tokenizer_ctx"))
    tokenizer_cross = RobertaTokenizer.from_pretrained(args.cross_encoder_path)

    bi_encoder = BiEncoder(model_name="bert-base-uncased", dropout=0.1)
    ckpt = torch.load(os.path.join(args.bi_encoder_path, args.bi_ckpt), map_location="cpu")
    bi_encoder.load_state_dict(ckpt["model_state"], strict=False)
    bi_encoder.query_encoder.resize_token_embeddings(len(tokenizer_q))
    bi_encoder.context_encoder.resize_token_embeddings(len(tokenizer_ctx))

    cross_encoder = CrossEncoder(model_name="roberta-base", dropout=0.1)
    ckpt_c = torch.load(os.path.join(args.cross_encoder_path, args.cross_ckpt), map_location="cpu")
    cross_encoder.load_state_dict(ckpt_c["model_state"], strict=False)
    cross_encoder.roberta.resize_token_embeddings(len(tokenizer_cross))

    device = torch.device(args.device)
    bi_encoder = bi_encoder.to(device)
    cross_encoder = cross_encoder.to(device)

    mkbr = MKBR(
        bi_encoder=bi_encoder,
        cross_encoder=cross_encoder,
        tokenizer_q=tokenizer_q,
        tokenizer_ctx=tokenizer_ctx,
        tokenizer_cross=tokenizer_cross,
        entity_base_dir=ENTITY_BASE_DIR,
        device=device,
        top_n_bi=args.top_n,
        top_k_final=max(fixed_k_list),
    )

    items = load_qa_split(QA_DATA_DIR, args.split)
    if args.max_eval:
        items = items[: args.max_eval]

    # per-item: full ranked list (serialized order), gold set
    # aggregate: recall for each fixed-k, recall for adaptive-k, avg k for adaptive
    results_fixed = {k: [] for k in fixed_k_list}
    results_adaptive_recall = []
    results_adaptive_k = []

    for item in tqdm(items, desc="Eval"):
        gold = get_gold_serialized(item)
        if not gold:
            continue
        table_id = item.get("table_id")
        if not table_id:
            continue
        try:
            ranked = mkbr.retrieve_ranked_with_scores(
                item["question"],
                table_id,
                TABLE_DIR,
                top_n=args.top_n,
            )
        except Exception as e:
            print(f"Error: {e}")
            continue
        if not ranked:
            continue
        serialized_list = [t["serialized"] for t in ranked]
        scores = np.array([t["score"] for t in ranked])

        for k in fixed_k_list:
            r = recall_at_k(serialized_list, gold, k)
            results_fixed[k].append(r)

        k_adaptive = adaptive_k_largest_gap(
            scores,
            top_frac=args.adaptive_top_frac,
            buffer=args.adaptive_buffer,
            min_k=1,
            max_k=len(ranked),
        )
        r_adaptive = recall_at_k(serialized_list, gold, k_adaptive)
        results_adaptive_recall.append(r_adaptive)
        results_adaptive_k.append(k_adaptive)

    n = len(results_adaptive_recall)
    if n == 0:
        print("No valid samples.")
        return

    # 표 구성: Method | Recall | (Adaptive-k만) Avg k
    rows = []
    for k in fixed_k_list:
        recs = results_fixed[k]
        if not recs:
            continue
        avg_r = np.mean(recs)
        rows.append({"method": f"Fixed-k={k}", "recall": round(float(avg_r), 4), "avg_k": k})

    avg_r_adaptive = np.mean(results_adaptive_recall)
    avg_k_adaptive = np.mean(results_adaptive_k)
    rows.append({
        "method": "Adaptive-k (ours)",
        "recall": round(float(avg_r_adaptive), 4),
        "avg_k": round(float(avg_k_adaptive), 2),
    })

    # 출력
    print("\n=== Fixed-k vs Adaptive-k (Recall) ===\n")
    print(f"{'Method':<25} {'Recall':>10} {'Avg k':>10}")
    print("-" * 47)
    for r in rows:
        print(f"{r['method']:<25} {r['recall']:>10.4f} {r['avg_k']:>10}")
    print("-" * 47)
    print(f"# samples: {n}")

    # 저장
    os.makedirs(args.output_dir, exist_ok=True)
    out_table_path = os.path.join(args.output_dir, "comparison_table.json")
    with open(out_table_path, "w", encoding="utf-8") as f:
        json.dump({"n_samples": n, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_table_path}")

    # CSV 저장 (표 비교용)
    import csv
    csv_path = os.path.join(args.output_dir, "comparison_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "recall", "avg_k"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
