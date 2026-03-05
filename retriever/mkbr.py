"""
Multistage KB Retriever (MKBR): Bi-encoder top-N -> Cross-encoder re-rank.
Inference: given question + table, retrieve top N triples (default 200), then re-rank and return top k.
"""

import torch
from typing import List, Optional

from . import serialization as ser
from .kb import build_subgraph_triples
from .dataset import get_table_entity_ids, load_table


class MKBR:
    def __init__(
        self,
        bi_encoder,
        cross_encoder,
        tokenizer_q,
        tokenizer_ctx,
        tokenizer_cross,
        entity_base_dir: str,
        device=None,
        top_n_bi: int = 200,
        top_k_final: int = 20,
        max_length_ctx: int = 256,
        max_length_cross: int = 512,
    ):
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self.tokenizer_q = tokenizer_q
        self.tokenizer_ctx = tokenizer_ctx
        self.tokenizer_cross = tokenizer_cross
        self.entity_base_dir = entity_base_dir
        self.device = device or next(bi_encoder.parameters()).device
        self.top_n_bi = top_n_bi
        self.top_k_final = top_k_final
        self.max_length_ctx = max_length_ctx
        self.max_length_cross = max_length_cross

    def get_triples_for_table(self, table_id: str, table_dir: str) -> list:
        table_obj = load_table(table_dir, table_id)
        if not table_obj:
            return []
        entity_ids = get_table_entity_ids(table_obj["data"])
        return build_subgraph_triples(entity_ids, self.entity_base_dir)

    def encode_question(self, question: str, batch_size: int = 32):
        self.bi_encoder.eval()
        enc = self.tokenizer_q(
            [question],
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            q_emb = self.bi_encoder.encode_query(enc["input_ids"], enc["attention_mask"])
        return q_emb

    def encode_contexts(self, contexts: List[str], batch_size: int = 32):
        self.bi_encoder.eval()
        all_emb = []
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i : i + batch_size]
            enc = self.tokenizer_ctx(
                batch,
                padding="max_length",
                max_length=self.max_length_ctx,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                c_emb = self.bi_encoder.encode_context(enc["input_ids"], enc["attention_mask"])
            all_emb.append(c_emb)
        return torch.cat(all_emb, dim=0)

    def retrieve(
        self,
        question: str,
        table_id: str,
        table_dir: str,
        top_n: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """
        Retrieve top-k triples for (question, table). Uses Bi-encoder to get top_n,
        then Cross-encoder to re-rank to top_k.
        Returns list of triple dicts with keys from build_subgraph_triples + "score".
        """
        top_n = top_n or self.top_n_bi
        top_k = top_k or self.top_k_final
        table_obj = load_table(table_dir, table_id)
        if not table_obj:
            return []
        header = table_obj["header"]
        data = table_obj["data"]
        triples = self.get_triples_for_table(table_id, table_dir)
        if not triples:
            return []
        contexts = []
        for t in triples:
            T_star = ser.table_to_serialized_with_subtable(header, data, t["head_id"])
            ctx = T_star + " " + t["serialized"]
            contexts.append(ctx)
        q_emb = self.encode_question(question).squeeze(0)  # (D,)
        c_emb = self.encode_contexts(contexts)  # (N, D)
        scores_bi = torch.matmul(c_emb, q_emb)  # (N,)
        _, indices = torch.topk(scores_bi, min(top_n, len(triples)))
        candidate_triples = [triples[i] for i in indices.cpu().tolist()]
        candidate_contexts = [contexts[i] for i in indices.cpu().tolist()]
        # Re-rank with cross-encoder
        cross_inputs = [question + " " + ctx for ctx in candidate_contexts]
        enc = self.tokenizer_cross(
            cross_inputs,
            padding="max_length",
            max_length=self.max_length_cross,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        self.cross_encoder.eval()
        with torch.no_grad():
            scores_cross = self.cross_encoder.score(enc["input_ids"], enc["attention_mask"])
        _, rerank_idx = torch.topk(scores_cross, min(top_k, len(candidate_triples)))
        out = []
        for i in rerank_idx.cpu().tolist():
            t = candidate_triples[i].copy()
            t["score"] = float(scores_cross[i].cpu())
            out.append(t)
        return out

    def retrieve_ranked_with_scores(
        self,
        question: str,
        table_id: str,
        table_dir: str,
        top_n: Optional[int] = None,
    ) -> List[dict]:
        """
        Bi-encoder로 top_n 후보를 뽑은 뒤 cross-encoder로 전부 re-rank하여,
        점수 기준 내림차순 전체 리스트를 반환. Adaptive-k 또는 임의의 k 평가용.
        각 항목: build_subgraph_triples 필드 + "score".
        """
        top_n = top_n or self.top_n_bi
        table_obj = load_table(table_dir, table_id)
        if not table_obj:
            return []
        header = table_obj["header"]
        data = table_obj["data"]
        triples = self.get_triples_for_table(table_id, table_dir)
        if not triples:
            return []
        contexts = []
        for t in triples:
            T_star = ser.table_to_serialized_with_subtable(header, data, t["head_id"])
            ctx = T_star + " " + t["serialized"]
            contexts.append(ctx)
        q_emb = self.encode_question(question).squeeze(0)
        c_emb = self.encode_contexts(contexts)
        scores_bi = torch.matmul(c_emb, q_emb)
        _, indices = torch.topk(scores_bi, min(top_n, len(triples)))
        candidate_triples = [triples[i] for i in indices.cpu().tolist()]
        candidate_contexts = [contexts[i] for i in indices.cpu().tolist()]
        cross_inputs = [question + " " + ctx for ctx in candidate_contexts]
        enc = self.tokenizer_cross(
            cross_inputs,
            padding="max_length",
            max_length=self.max_length_cross,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        self.cross_encoder.eval()
        with torch.no_grad():
            scores_cross = self.cross_encoder.score(enc["input_ids"], enc["attention_mask"])
        scores_np = scores_cross.cpu().numpy()
        order = scores_np.argsort()[::-1]
        out = []
        for i in order:
            t = candidate_triples[i].copy()
            t["score"] = float(scores_np[i])
            out.append(t)
        return out
