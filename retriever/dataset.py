"""
KET-QA retrieval dataset: load QA items, build sub-graph triples per table,
positive = gold evidence (In KB), negative = kNN or random from non-gold.
"""

import json
import os
from typing import Callable

import torch
from torch.utils.data import Dataset

from . import serialization as ser
from .kb import build_subgraph_triples, evidence_to_triple_dict, _load_entity


def get_table_entity_ids(table_data: list) -> set:
    """Collect all entity IDs from table data (list of rows)."""
    entity_ids = set()
    for row in table_data:
        entity_ids |= ser.get_entity_ids_from_row(row)
    return entity_ids


def load_table(table_dir: str, table_id: str) -> dict | None:
    path = os.path.join(table_dir, f"{table_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_qa_split(data_dir: str, split: str) -> list:
    path = os.path.join(data_dir, f"{split}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RetrievalDataset(Dataset):
    """
    Dataset for retriever training. Each item = one (question, table, triple, label).
    For Bi-encoder: we need (question, context = T*_sub ⊕ t*, label).
    For Cross-encoder: we need (question ⊕ T*_sub ⊕ t*, label).
    """

    def __init__(
        self,
        qa_items: list,
        table_dir: str,
        entity_base_dir: str,
        split: str,
        negative_sampling: str = "random",  # "random" | "knn" (knn은 별도 precompute 필요, 현재는 random만 동작)
        n_negatives: int = 25,
        max_triples_per_table: int = 5000,
        tokenizer_question=None,
        tokenizer_context=None,
        max_length_q: int = 128,
        max_length_ctx: int = 256,
        for_cross_encoder: bool = False,
        sentence_encoder_for_knn=None,  # kNS 사용 시: question/context 임베딩해 kNN negative 뽑는 encoder
        device=None,
    ):
        self.qa_items = qa_items
        self.table_dir = table_dir
        self.entity_base_dir = entity_base_dir
        self.split = split
        self.negative_sampling = negative_sampling
        self.n_negatives = n_negatives
        self.max_triples_per_table = max_triples_per_table
        self.tokenizer_question = tokenizer_question
        self.tokenizer_context = tokenizer_context
        self.max_length_q = max_length_q
        self.max_length_ctx = max_length_ctx
        self.for_cross_encoder = for_cross_encoder
        self.sentence_encoder_for_knn = sentence_encoder_for_knn
        self.device = device

        # Precompute per (table_id, question_id): table, triples, gold set
        self._samples = []  # list of (q, table_serialized_per_triple, triple_serialized, head_id, label)
        self._build_samples()

    def _build_samples(self):
        from tqdm import tqdm
        for item in tqdm(self.qa_items, desc=f"Building {self.split} retrieval dataset"):
            table_id = item.get("table_id")
            if not table_id:
                continue
            table_obj = load_table(self.table_dir, table_id)
            if not table_obj:
                continue
            header = table_obj.get("header", [])
            data = table_obj.get("data", [])
            entity_ids = get_table_entity_ids(data)
            triples = build_subgraph_triples(entity_ids, self.entity_base_dir)
            if len(triples) > self.max_triples_per_table:
                triples = triples[: self.max_triples_per_table]
            gold_evidence = item.get("selected_evidence_candidate", [])
            gold_triples = set()
            for ev in gold_evidence:
                td = evidence_to_triple_dict(ev)
                if td:
                    gold_triples.add(td["serialized"])
            question = item.get("question", "")
            # Sub-table per triple: use head_id for row filtering
            for t in triples:
                head_id = t["head_id"]
                t_star = t["serialized"]
                T_star = ser.table_to_serialized_with_subtable(header, data, head_id)
                context = T_star + " " + t_star
                label = 1 if t_star in gold_triples else 0
                self._samples.append({
                    "question": question,
                    "table_id": table_id,
                    "context": context,
                    "triple_serialized": t_star,
                    "label": label,
                })
        # For training we need (q, pos_contexts, neg_contexts). So we group by (question, table) and sample negatives.
        # Simpler: keep flat list; when sampling batch we have multiple rows per question with label 0/1.
        # So each row is (q, context, label). For contrastive we need in one batch: one q, one positive, n negatives.
        # So we need to group by question+table and yield one positive and n negatives per step. That requires a different sampler or dataset design.
        # Paper: "The i-th instance contains one question qi, one table T, m relevant (positive) triples and n irrelevant (negative) triples."
        # So one training sample = (qi, Ti, t+_1..t+_p, t-_1..t-_n). We'll sample these in __getitem__ by grouping on the fly or pre-group.
        # Pre-group: build list of groups (question, table_id, positive_triples_list, all_triples_list). Then __getitem__(i) returns group i; collate will form (q, T*, pos_contexts, neg_contexts) and sample n_neg from neg_contexts.
        # Actually the paper says contrastive loss: for each positive t+_j, loss = -log( exp(s(t+)) / (exp(s(t+)) + sum(exp(s(t-_k)))). So we need one positive and n negatives per step. So we can:
        # Option A: Each item = (q, one positive context, list of n negative contexts). Then we need to precompute for each (q, table) one positive and n negatives.
        # Option B: Each item = (q, context, label). Sample batches such that we have at least one positive per question; use in-batch negatives (other contexts in batch as negatives). That's another common approach.
        # Paper uses explicit kNN negatives. So we need to precompute for each (q, table): positives, and n negatives (kNN from non-positives). So we need to go back and build grouped samples.
        self._grouped = self._group_by_question_table()

    def _group_by_question_table(self):
        from collections import defaultdict
        groups = defaultdict(lambda: {"question": "", "table_id": "", "pos": [], "neg": []})
        for s in self._samples:
            key = (s["question"], s["table_id"])
            groups[key]["question"] = s["question"]
            groups[key]["table_id"] = s["table_id"]
            if s["label"] == 1:
                groups[key]["pos"].append(s["context"])
            else:
                groups[key]["neg"].append(s["context"])
        # Filter to groups that have at least one positive
        return [g for g in groups.values() if g["pos"]]

    def __len__(self):
        return len(self._grouped)

    def __getitem__(self, idx):
        g = self._grouped[idx]
        q = g["question"]
        pos_list = g["pos"]
        neg_list = g["neg"]
        # Sample one positive and n_negatives negatives.
        # negative_sampling=="knn"이면 논문처럼 질문에 가까운 hard negative를 써야 하나,
        # 현재는 random만 구현됨. kNS 쓰려면 별도로 encoder로 질문/neg 임베딩 후 kNN 선택해 두고 로드해야 함.
        import random
        pos = random.choice(pos_list)
        if len(neg_list) >= self.n_negatives:
            negs = random.sample(neg_list, self.n_negatives)
        else:
            # Repeat negatives if not enough; if no negs at all use positive as placeholder
            negs = (neg_list * (self.n_negatives // max(1, len(neg_list)) + 1))[: self.n_negatives] if neg_list else [pos] * self.n_negatives
        if self.for_cross_encoder:
            # Cross: input = q ⊕ T* ⊕ t*
            return {
                "question": q,
                "positive_context": pos,
                "negative_contexts": negs,
            }
        return {
            "question": q,
            "positive_context": pos,
            "negative_contexts": negs,
        }


def collate_bi_encoder(batch, tokenizer_q, tokenizer_ctx, max_len_q, max_len_ctx):
    """Collate so we have: query_ids, query_mask, ctx_ids (pos + neg), ctx_mask."""
    questions = [b["question"] for b in batch]
    pos_ctx = [b["positive_context"] for b in batch]
    neg_ctxs = [b["negative_contexts"] for b in batch]
    q_enc = tokenizer_q(
        questions,
        padding="max_length",
        max_length=max_len_q,
        truncation=True,
        return_tensors="pt",
    )
    all_ctx = []
    for i, negs in enumerate(neg_ctxs):
        all_ctx.append(pos_ctx[i])
        all_ctx.extend(negs)
    ctx_enc = tokenizer_ctx(
        all_ctx,
        padding="max_length",
        max_length=max_len_ctx,
        truncation=True,
        return_tensors="pt",
    )
    return {
        "query_input_ids": q_enc["input_ids"],
        "query_attention_mask": q_enc["attention_mask"],
        "context_input_ids": ctx_enc["input_ids"],
        "context_attention_mask": ctx_enc["attention_mask"],
        "n_pos": 1,
        "n_neg": len(neg_ctxs[0]),
    }


def collate_cross_encoder(batch, tokenizer, max_length: int = 512):
    """Collate for cross-encoder: input = question + context (T* ⊕ t*)."""
    pos_inputs = [b["question"] + " " + b["positive_context"] for b in batch]
    neg_batches = []
    for b in batch:
        for neg in b["negative_contexts"]:
            neg_batches.append(b["question"] + " " + neg)
    all_inputs = pos_inputs + neg_batches
    labels = [1] * len(pos_inputs) + [0] * len(neg_batches)
    enc = tokenizer(
        all_inputs,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
    }
