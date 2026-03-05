#!/usr/bin/env python3
"""
Train Bi-encoder for KET-QA Retriever.
Paper: 20 epochs, batch 16, kNS n=25, AdamW 1e-5, linear schedule, warm-up, dropout 0.1.
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever.config import (
    set_data_paths,
    LEARNING_RATE,
    WARMUP_RATIO,
    DROPOUT_RATE,
    ADAMW_WEIGHT_DECAY,
    BI_ENCODER_EPOCHS,
    BI_ENCODER_BATCH_SIZE,
    BI_ENCODER_N_NEGATIVES,
    BI_ENCODER_MODEL,
)
from retriever.dataset import load_qa_split, RetrievalDataset, collate_bi_encoder
from retriever.bi_encoder import BiEncoder, contrastive_loss
from retriever.serialization import HEAD_TOKEN, REL_TOKEN, TAIL_TOKEN, ROW_TOKEN


def add_special_tokens(tokenizer):
    special = [HEAD_TOKEN, REL_TOKEN, TAIL_TOKEN, ROW_TOKEN]
    added = tokenizer.add_special_tokens({"additional_special_tokens": special})
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset_ketqa")
    parser.add_argument("--output_dir", type=str, default="outputs/bi_encoder")
    parser.add_argument("--model_name", type=str, default=BI_ENCODER_MODEL)
    parser.add_argument("--epochs", type=int, default=BI_ENCODER_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BI_ENCODER_BATCH_SIZE,
                        help="맥북 MPS 18GB: 2 권장. 16GB 램: 4~8")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--n_negatives", type=int, default=BI_ENCODER_N_NEGATIVES,
                        help="맥북 MPS OOM 시 10 등으로 줄이기")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation. effective_batch = batch_size * accumulation_steps (논문 16 맞추기용)")
    parser.add_argument("--max_length_q", type=int, default=128)
    parser.add_argument("--max_length_ctx", type=int, default=256)
    parser.add_argument("--max_triples_per_table", type=int, default=3000)
    parser.add_argument("--max_train_items", type=int, default=None,
                        help="Train 샘플 수 상한 (train.json 앞에서 N개만 사용). 예: 500 = (question, table) 500개로만 학습")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda | mps(Mac GPU) | cpu. 기본: cuda > mps > cpu")
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

    set_data_paths(args.data_root)
    from retriever.config import TABLE_DIR, ENTITY_BASE_DIR, QA_DATA_DIR

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_q = BertTokenizer.from_pretrained(args.model_name)
    tokenizer_ctx = BertTokenizer.from_pretrained(args.model_name)
    add_special_tokens(tokenizer_q)
    add_special_tokens(tokenizer_ctx)

    train_items = load_qa_split(QA_DATA_DIR, "train")
    if args.max_train_items:
        train_items = train_items[: args.max_train_items]
    dev_items = load_qa_split(QA_DATA_DIR, "dev")

    train_ds = RetrievalDataset(
        train_items,
        TABLE_DIR,
        ENTITY_BASE_DIR,
        "train",
        negative_sampling="random",
        n_negatives=args.n_negatives,
        max_triples_per_table=args.max_triples_per_table,
        for_cross_encoder=False,
    )
    dev_ds = RetrievalDataset(
        dev_items,
        TABLE_DIR,
        ENTITY_BASE_DIR,
        "dev",
        negative_sampling="random",
        n_negatives=args.n_negatives,
        max_triples_per_table=args.max_triples_per_table,
        for_cross_encoder=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_bi_encoder(
            b, tokenizer_q, tokenizer_ctx, args.max_length_q, args.max_length_ctx
        ),
    )

    device = torch.device(args.device)
    model = BiEncoder(model_name=args.model_name, dropout=DROPOUT_RATE)
    model.query_encoder.resize_token_embeddings(len(tokenizer_q))
    model.context_encoder.resize_token_embeddings(len(tokenizer_ctx))
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=ADAMW_WEIGHT_DECAY)
    steps_per_epoch = (len(train_loader) + args.accumulation_steps - 1) // args.accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        for i, batch in enumerate(pbar):
            q_ids = batch["query_input_ids"].to(device)
            q_mask = batch["query_attention_mask"].to(device)
            ctx_ids = batch["context_input_ids"].to(device)
            ctx_mask = batch["context_attention_mask"].to(device)
            n_pos = batch["n_pos"]
            n_neg = batch["n_neg"]

            q_emb, c_emb = model(q_ids, q_mask, ctx_ids, ctx_mask)
            B = q_emb.size(0)
            c_emb = c_emb.view(B, 1 + n_neg, -1)
            loss = contrastive_loss(q_emb, c_emb, n_pos=1, n_neg=n_neg) / args.accumulation_steps
            loss.backward()
            total_loss += loss.item() * args.accumulation_steps

            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            pbar.set_postfix(loss=loss.item() * args.accumulation_steps)
        print(f"Epoch {epoch+1} avg loss: {total_loss / len(train_loader):.4f}")

        save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "epoch": epoch + 1,
        }, save_path)

    tokenizer_q.save_pretrained(args.output_dir)
    tokenizer_ctx.save_pretrained(os.path.join(args.output_dir, "tokenizer_ctx"))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
