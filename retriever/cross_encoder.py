"""
Re-Rank Cross-Encoder: 1 RoBERTa, input = q ⊕ T* ⊕ t*, binary classification (logit as score).
s(t, q, T) = E(q ⊕ T* ⊕ t*)
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer


class CrossEncoder(nn.Module):
    """
    Single RoBERTa takes concatenated [question, table, triple]. [CLS] logit for relevance (0/1).
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.1,
        num_labels: int = 2,
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
    ):
        out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = self.classifier(cls)  # (B, 2)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    def score(self, input_ids, attention_mask):
        """Return relevance score (logit of positive class)."""
        out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]  # (B, 2)
        return logits[:, 1]  # positive class logit
