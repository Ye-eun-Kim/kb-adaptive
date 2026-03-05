#!/usr/bin/env python3
"""
Evaluate KET-QA Retriever with Recall@k.
R@k = (1/N) * sum_i (|evidence retrieved|_i / |gold evidence|_i)
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever.config import set_data_paths
from retriever.dataset import load_qa_split, load_table, get_table_entity_ids
from retriever.kb import build_subgraph_triples, evidence_to_triple_dict
from retriever import serialization as ser
from retriever.mkbr import MKBR
from retriever.bi_encoder import BiEncoder
from retriever.cross_encoder import CrossEncoder
from transformers import BertTokenizer, RobertaTokenizer


def get_gold_serialized(item) -> set:
    gold = set()
    for ev in item.get("selected_evidence_candidate", []):
        td = evidence_to_triple_dict(ev)
        if td:
            gold.add(td["serialized"])
    return gold


def recall_at_k(retrieved_serialized: list, gold: set, k: int) -> float:
    """|retrieved ∩ gold| up to k / |gold|."""
    if not gold:
        return 1.0
    top_k = retrieved_serialized[:k]
    hit = len(gold & set(top_k))
    return hit / len(gold)


def _resolve_ckpt(dir_path: str, filename: str) -> str:
    """지정한 파일이 없으면 해당 디렉터리에서 epoch_*.pt 중 가장 최신 파일 사용."""
    path = os.path.join(dir_path, filename)
    if os.path.isfile(path):
        return filename
    import glob
    candidates = glob.glob(os.path.join(dir_path, "epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {dir_path}. Expected {filename} or epoch_*.pt")
    latest = max(candidates, key=lambda p: int(p.replace(".pt", "").split("epoch_")[-1]))
    return os.path.basename(latest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset_ketqa")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--bi_encoder_path", type=str, default="outputs/bi_encoder")
    parser.add_argument("--cross_encoder_path", type=str, default="outputs/cross_encoder")
    parser.add_argument("--bi_ckpt", type=str, default="epoch_20.pt",
                        help="Bi-encoder 체크포인트 파일명. 없으면 해당 디렉터리에서 최신 epoch_*.pt 자동 선택")
    parser.add_argument("--cross_ckpt", type=str, default="epoch_5.pt",
                        help="Cross-encoder 체크포인트 파일명. 없으면 해당 디렉터리에서 최신 epoch_*.pt 자동 선택")
    parser.add_argument("--top_n", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_eval", type=int, default=None,
                        help="평가할 샘플 수 상한. 미지정 시 전체 데이터로 평가")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_data_paths(args.data_root)
    from retriever.config import TABLE_DIR, ENTITY_BASE_DIR, QA_DATA_DIR

    bi_ckpt = _resolve_ckpt(args.bi_encoder_path, args.bi_ckpt)
    cross_ckpt = _resolve_ckpt(args.cross_encoder_path, args.cross_ckpt)
    print(f"Bi-encoder checkpoint: {bi_ckpt}, Cross-encoder checkpoint: {cross_ckpt}")

    tokenizer_q = BertTokenizer.from_pretrained(args.bi_encoder_path)
    tokenizer_ctx = BertTokenizer.from_pretrained(os.path.join(args.bi_encoder_path, "tokenizer_ctx"))
    tokenizer_cross = RobertaTokenizer.from_pretrained(args.cross_encoder_path)

    # 체크포인트는 학습 시 추가한 special token 포함 vocab 크기로 저장됨 → 로드 전에 resize 필요
    bi_encoder = BiEncoder(model_name="bert-base-uncased", dropout=0.1)
    bi_encoder.query_encoder.resize_token_embeddings(len(tokenizer_q))
    bi_encoder.context_encoder.resize_token_embeddings(len(tokenizer_ctx))
    ckpt = torch.load(os.path.join(args.bi_encoder_path, bi_ckpt), map_location="cpu")
    bi_encoder.load_state_dict(ckpt["model_state"], strict=False)

    cross_encoder = CrossEncoder(model_name="roberta-base", dropout=0.1)
    cross_encoder.roberta.resize_token_embeddings(len(tokenizer_cross))
    ckpt_c = torch.load(os.path.join(args.cross_encoder_path, cross_ckpt), map_location="cpu")
    cross_encoder.load_state_dict(ckpt_c["model_state"], strict=False)

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
        top_k_final=args.top_k,
    )

    items = load_qa_split(QA_DATA_DIR, args.split)
    if args.max_eval is not None:
        items = items[: args.max_eval]
        print(f"Evaluating on first {len(items)} samples (--max_eval={args.max_eval})")
    else:
        print(f"Evaluating on full {args.split} set ({len(items)} samples)")

    r1, r5, r20, r100 = 0.0, 0.0, 0.0, 0.0
    n = 0
    for item in tqdm(items, desc="Eval"):
        gold = get_gold_serialized(item)
        if not gold:
            continue
        table_id = item.get("table_id")
        if not table_id:
            continue
        try:
            retrieved = mkbr.retrieve(
                item["question"],
                table_id,
                TABLE_DIR,
                top_n=args.top_n,
                top_k=min(100, args.top_n),
            )
        except Exception as e:
            print(f"Error: {e}")
            continue
        ret_serialized = [t["serialized"] for t in retrieved]
        r1 += recall_at_k(ret_serialized, gold, 1)
        r5 += recall_at_k(ret_serialized, gold, 5)
        r20 += recall_at_k(ret_serialized, gold, 20)
        r100 += recall_at_k(ret_serialized, gold, 100)
        n += 1

    if n == 0:
        print("No valid samples.")
        return
    n = float(n)
    print(f"Recall@1:  {r1/n:.4f}")
    print(f"Recall@5:  {r5/n:.4f}")
    print(f"Recall@20: {r20/n:.4f}")
    print(f"Recall@100: {r100/n:.4f}")


if __name__ == "__main__":
    main()
