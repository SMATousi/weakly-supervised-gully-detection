"""Student heads on frozen tokens x: (B, 8, T, D) fp16/fp32, valid: (B, 8) bool. Output: logits (B, 2)."""
from __future__ import annotations

import torch
from torch import nn


class LinearHead(nn.Module):
    """Legacy head: flatten all tokens -> LayerNorm -> BatchNorm -> Dropout -> one Linear layer."""

    def __init__(self, n_img: int = 8, tokens: int = 65, dim: int = 384, dropout: float = 0.3):
        super().__init__()
        d = n_img * tokens * dim
        self.norm, self.bn, self.drop = nn.LayerNorm(d), nn.BatchNorm1d(d), nn.Dropout(dropout)
        self.fc = nn.Linear(d, 2)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, valid=None):
        x = x.float().flatten(1)
        return self.fc(self.drop(self.bn(self.norm(x))))


class MeanPoolMLP(nn.Module):
    """Per image [CLS, mean of patch tokens] -> concat over the 8 images -> MLP."""

    def __init__(self, n_img: int = 8, dim: int = 384, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(n_img * 2 * dim), nn.Dropout(dropout),
                                 nn.Linear(n_img * 2 * dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 2))

    def forward(self, x, valid=None):
        x = x.float()
        f = torch.cat([x[:, :, 0], x[:, :, 1:].mean(2)], -1)  # (B, 8, 2D)
        if valid is not None:
            f = f * valid[..., None]
        return self.net(f.flatten(1))


class TemporalAttentionHead(nn.Module):
    """Project tokens, add date/role and within-image position embeddings, 2-layer transformer over all
    8 x T tokens plus a learned [GLOBAL] token; masked for invalid images."""

    def __init__(self, n_img: int = 8, tokens: int = 65, dim: int = 384, d_model: int = 256,
                 layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d_model))
        self.date = nn.Parameter(torch.zeros(n_img, 1, d_model))  # date (NAIP bins) and resolution (high-res) role
        self.pos = nn.Parameter(torch.zeros(1, tokens, d_model))
        self.glob = nn.Parameter(torch.zeros(1, 1, d_model))
        for p in (self.date, self.pos, self.glob):
            nn.init.trunc_normal_(p, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, x, valid=None):
        b, _, t, _ = x.shape
        h = self.proj(x.float()) + self.date + self.pos  # (B, 8, T, d)
        h = torch.cat([self.glob.expand(b, -1, -1), h.flatten(1, 2)], 1)
        mask = None
        if valid is not None:
            mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device),
                              (~valid).repeat_interleave(t, 1)], 1)
        return self.out(self.enc(h, src_key_padding_mask=mask)[:, 0])


HEADS = {"linear": LinearHead, "mlp": MeanPoolMLP, "temporal": TemporalAttentionHead}
