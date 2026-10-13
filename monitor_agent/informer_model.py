"""
Informer transformer for time‑series anomaly prediction.
Based on Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time‑Series Forecasting" (AAAI 2021).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class ProbSparseAttention(nn.Module):
    """
    ProbSparse attention mechanism: selects the most "active" queries for efficiency.
    """
    def __init__(self, d_model: int, n_heads: int, factor: int = 5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.d_k = d_model // n_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_Q, _ = queries.shape
        B, L_KV, _ = keys.shape

        # Reshape for multi‑head attention
        queries = queries.view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        keys = keys.view(B, L_KV, self.n_heads, self.d_k).transpose(1, 2)
        values = values.view(B, L_KV, self.n_heads, self.d_k).transpose(1, 2)

        # ProbSparse: sample keys to compute approximate attention
        U = self.factor * int(math.log(L_KV))
        sample_keys = keys[:, :, torch.randperm(L_KV)[:U], :]

        # Compute scores for sampled keys
        scores = torch.matmul(queries, sample_keys.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Identify top‑u queries with highest variance
        M = scores.max(dim=-1)[0] - torch.mean(scores, dim=-1)
        M_top = torch.topk(M, int(self.factor * math.log(L_Q)), dim=-1)[1]

        # Full attention only for selected queries
        attn = torch.zeros(B, self.n_heads, L_Q, L_KV, device=queries.device)
        attn[:, :, M_top, :] = F.softmax(
            torch.matmul(queries[:, :, M_top, :], keys.transpose(-2, -1)) / math.sqrt(self.d_k),
            dim=-1
        )

        # For other queries, use uniform distribution
        attn.fill_(1.0 / L_KV)

        output = torch.matmul(attn, values)
        output = output.transpose(1, 2).contiguous().view(B, L_Q, self.d_model)

        return output, attn

class EncoderLayer(nn.Module):
    """Single encoder layer with ProbSparse attention and feed‑forward."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class Informer(nn.Module):
    """
    Informer transformer for time‑series anomaly detection.
    """
    def __init__(self, n_steps: int, n_features: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 3, d_ff: int = 512, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.d_model = d_model

        # Input embedding
        self.embedding = nn.Linear(n_features, d_model)
        self.position_encoding = PositionalEncoding(d_model, dropout)

        # Encoder layers
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, factor, dropout)
            for _ in range(n_layers)
        ])

        # Prediction heads for each horizon
        self.prediction_heads = nn.ModuleDict({
            f"horizon_{h}": nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)  # temperature prediction
            )
            for h in [30, 60, 90]
        })

        # Uncertainty (quantile) output
        self.quantile_heads = nn.ModuleDict({
            f"quantile_{h}": nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Linear(32, 3)  # 0.1, 0.5, 0.9 quantiles
            )
            for h in [30, 60, 90]
        })

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (batch, n_steps, n_features)
        Returns: dict with predictions for each horizon
        """
        x = self.embedding(x)
        x = self.position_encoding(x)

        for layer in self.encoder:
            x = layer(x)

        context = x[:, -1, :]  # (batch, d_model)

        predictions = {}
        uncertainties = {}

        for h in [30, 60, 90]:
            pred = self.prediction_heads[f"horizon_{h}"](context)
            predictions[h] = pred.squeeze(-1)

            quantiles = self.quantile_heads[f"quantile_{h}"](context)
            uncertainties[h] = {
                "q10": quantiles[:, 0],
                "q50": quantiles[:, 1],
                "q90": quantiles[:, 2]
            }

        return {"predictions": predictions, "uncertainties": uncertainties}

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)