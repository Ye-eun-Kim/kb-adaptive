"""
Retrieval Bi-Encoder: 2 BERT, dot product similarity, contrastive loss.
s(t, q, T) = Eq(q)^T Ec(T* ⊕ t*)
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig


class BiEncoder(nn.Module):
    """
    Two independent BERT encoders. Query encoder Eq for question, context encoder Ec
    for (T* ⊕ t*). Score = dot product of [CLS] vectors.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
        projection_dim: int = 0,  # 0 = use hidden size as-is
    ):
        super().__init__()
        self.query_encoder = BertModel.from_pretrained(model_name)
        self.context_encoder = BertModel.from_pretrained(model_name)
        hidden = self.query_encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.projection_dim = projection_dim
        if projection_dim and projection_dim != hidden:
            self.query_proj = nn.Linear(hidden, projection_dim)
            self.ctx_proj = nn.Linear(hidden, projection_dim)
        else:
            self.query_proj = self.ctx_proj = None

    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        context_input_ids,
        context_attention_mask,
    ):
        q_out = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
        )
        c_out = self.context_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
        )
        q_cls = q_out.last_hidden_state[:, 0, :]  # (B, H)
        c_cls = c_out.last_hidden_state[:, 0, :]  # (B*M, H)
        q_cls = self.dropout(q_cls)
        c_cls = self.dropout(c_cls)
        if self.query_proj is not None:
            q_cls = self.query_proj(q_cls)
            c_cls = self.ctx_proj(c_cls)
        return q_cls, c_cls

    def encode_query(self, input_ids, attention_mask):
        out = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
        q = out.last_hidden_state[:, 0, :]
        if self.query_proj is not None:
            q = self.query_proj(q)
        return self.dropout(q)

    def encode_context(self, input_ids, attention_mask):
        out = self.context_encoder(input_ids=input_ids, attention_mask=attention_mask)
        c = out.last_hidden_state[:, 0, :]
        if self.ctx_proj is not None:
            c = self.ctx_proj(c)
        return self.dropout(c)


def contrastive_loss(
    q_emb: torch.Tensor,  # (B, D)
    c_emb: torch.Tensor,  # (B, 1+n_neg, D) or (B*(1+n_neg), D)
    n_pos: int = 1,
    n_neg: int = 25,
) -> torch.Tensor:
    """
    Contrastive loss from paper: for each of m positives
    L = -log( exp(s(t+)) / (exp(s(t+)) + sum_k exp(s(t-_k))) )
    We have one positive and n negatives per query. So c_emb is (B, 1+n_neg, D).
    Score pos = dot(q, c_pos), scores neg = dot(q, c_neg_k).
    """
    if c_emb.dim() == 3:
        # (B, 1+n_neg, D)
        scores = torch.bmm(
            q_emb.unsqueeze(1),
            c_emb.transpose(1, 2),
        ).squeeze(1)  # (B, 1+n_neg)
    else:
        # (B*(1+n_neg), D) -> reshape and compute per-query
        B = q_emb.size(0)
        M = c_emb.size(0) // B
        c_emb = c_emb.view(B, M, -1)
        scores = torch.bmm(q_emb.unsqueeze(1), c_emb.transpose(1, 2)).squeeze(1)
    # scores[:, 0] = positive, scores[:, 1:] = negatives
    pos_scores = scores[:, 0]  # (B,)
    neg_scores = scores[:, 1:]  # (B, n_neg)
    # log softmax over (pos, neg_1, ..., neg_n)
    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # (B, 1+n_neg)
    log_probs = torch.nn.functional.log_softmax(all_scores, dim=1)
    loss = -log_probs[:, 0].mean()
    return loss
