"""Attention MIL survival model."""

from __future__ import annotations


class GatedAttentionMIL:
    """Gated attention pooling from Ilse et al. style MIL."""

    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        if cls is GatedAttentionMIL:

            class _GatedAttentionMIL(nn.Module):
                def __init__(self, input_dim: int, attention_dim: int = 256, dropout: float = 0.1):
                    import torch  # noqa: PLC0415

                    super().__init__()
                    self._torch = torch
                    self.attention_v = nn.Sequential(
                        nn.Linear(input_dim, attention_dim),
                        nn.Tanh(),
                        nn.Dropout(dropout),
                    )
                    self.attention_u = nn.Sequential(
                        nn.Linear(input_dim, attention_dim),
                        nn.Sigmoid(),
                        nn.Dropout(dropout),
                    )
                    self.attention = nn.Linear(attention_dim, 1)

                def forward(self, features, mask=None):
                    torch = self._torch
                    scores = self.attention(
                        self.attention_v(features) * self.attention_u(features)
                    ).squeeze(-1)
                    if mask is not None:
                        valid = mask.bool()
                        scores = scores.masked_fill(~valid, -torch.finfo(scores.dtype).max)

                    weights = torch.softmax(scores, dim=1)
                    if mask is not None:
                        weights = torch.where(valid, weights, torch.zeros_like(weights))
                        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

                    pooled = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
                    return pooled, weights

            return _GatedAttentionMIL(*args, **kwargs)
        return super().__new__(cls)


class CoxMILSurvivalModel:
    """Attention MIL model that predicts one Cox risk score per slide."""

    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        if cls is CoxMILSurvivalModel:

            class _CoxMILSurvivalModel(nn.Module):
                def __init__(
                    self,
                    feature_dim: int,
                    clinical_dim: int = 0,
                    attention_dim: int = 256,
                    hidden_dim: int = 256,
                    dropout: float = 0.15,
                ):
                    import torch  # noqa: PLC0415

                    super().__init__()
                    self._torch = torch
                    self.feature_norm = nn.LayerNorm(feature_dim)
                    self.mil = GatedAttentionMIL(feature_dim, attention_dim, dropout)
                    self.clinical_dim = clinical_dim

                    if clinical_dim:
                        self.clinical_net = nn.Sequential(
                            nn.LayerNorm(clinical_dim),
                            nn.Linear(clinical_dim, hidden_dim // 2),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        )
                        head_input_dim = feature_dim + hidden_dim // 2
                    else:
                        self.clinical_net = None
                        head_input_dim = feature_dim

                    self.head = nn.Sequential(
                        nn.Linear(head_input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1),
                    )

                def forward(self, features, mask=None, clinical=None):
                    torch = self._torch
                    features = self.feature_norm(features)
                    pooled, attention = self.mil(features, mask)

                    if self.clinical_net is not None:
                        if clinical is None:
                            raise ValueError("clinical covariates are required by this model")
                        clinical_embedding = self.clinical_net(clinical)
                        pooled = torch.cat([pooled, clinical_embedding], dim=-1)

                    risk = self.head(pooled).squeeze(-1)
                    return {"risk": risk, "attention": attention}

            return _CoxMILSurvivalModel(*args, **kwargs)
        return super().__new__(cls)
