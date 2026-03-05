#!/usr/bin/env python3
"""
Run MKBR retrieval for a single (question, table_id) or over a split.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever.config import set_data_paths
from retriever.mkbr import MKBR
from retriever.bi_encoder import BiEncoder
from retriever.cross_encoder import CrossEncoder
from transformers import BertTokenizer, RobertaTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset_ketqa")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--table_id", type=str, default=None)
    parser.add_argument("--bi_encoder_path", type=str, default="outputs/bi_encoder")
    parser.add_argument("--cross_encoder_path", type=str, default="outputs/cross_encoder")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_data_paths(args.data_root)
    from retriever.config import TABLE_DIR, ENTITY_BASE_DIR

    tokenizer_q = BertTokenizer.from_pretrained(args.bi_encoder_path)
    tokenizer_ctx = BertTokenizer.from_pretrained(os.path.join(args.bi_encoder_path, "tokenizer_ctx"))
    tokenizer_cross = RobertaTokenizer.from_pretrained(args.cross_encoder_path)

    bi_encoder = BiEncoder(model_name="bert-base-uncased", dropout=0.1)
    ckpt = torch.load(os.path.join(args.bi_encoder_path, "epoch_20.pt"), map_location="cpu")
    bi_encoder.load_state_dict(ckpt["model_state"], strict=False)
    bi_encoder.query_encoder.resize_token_embeddings(len(tokenizer_q))
    bi_encoder.context_encoder.resize_token_embeddings(len(tokenizer_ctx))

    cross_encoder = CrossEncoder(model_name="roberta-base", dropout=0.1)
    ckpt_c = torch.load(os.path.join(args.cross_encoder_path, "epoch_5.pt"), map_location="cpu")
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
        top_n_bi=200,
        top_k_final=args.top_k,
    )

    if args.question and args.table_id:
        results = mkbr.retrieve(args.question, args.table_id, TABLE_DIR, top_k=args.top_k)
        for i, t in enumerate(results):
            print(f"{i+1}. [{t['score']:.3f}] {t['serialized']}")
    else:
        print("Use --question and --table_id for single query. Example:")
        print('  python run_retrieval.py --question "What was the release date of the studio album?" --table_id "Gene_Tierney_1"')


if __name__ == "__main__":
    main()
