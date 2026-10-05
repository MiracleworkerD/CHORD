from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScatteredPadder(nn.Module):
    def __init__(self, max_dim: int, scattered: bool = True):
        super().__init__()
        self.max_dim = max_dim
        self.scattered = scattered
        self._cache = {}

    def _positions(self, h: int, device: torch.device) -> torch.Tensor:
        key = (h, str(device))
        if key not in self._cache:
            idx = torch.arange(h, device=device, dtype=torch.float32)
            pos = torch.floor(idx * self.max_dim / h).long()
            self._cache[key] = pos.clamp_(0, self.max_dim - 1)
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, l, h = x.shape
        if h == self.max_dim:
            return x
        if h > self.max_dim:
            return x[..., : self.max_dim]
        out = x.new_zeros(b, t, l, self.max_dim)
        if not self.scattered:
            out[..., :h] = x
            return out
        pos = self._positions(h, x.device)
        return out.scatter(-1, pos.view(1, 1, 1, h).expand(b, t, l, h), x)


class EdgeRelationBias(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_tokens: int,
        num_blocks: int,
        rel_pos_dim: int,
        num_heads: int,
        dropout: float,
        use_cosine: bool,
        use_distance: bool,
        edge_bias_clip: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.use_cosine = use_cosine
        self.use_distance = use_distance
        self.edge_bias_clip = edge_bias_clip
        self.layer_rel = nn.Embedding(2 * num_layers - 1, rel_pos_dim)
        self.token_rel = nn.Embedding(2 * num_tokens - 1, rel_pos_dim)
        self.block_rel = nn.Embedding(2 * num_blocks - 1, rel_pos_dim)
        scalar_dim = int(use_cosine) + int(use_distance)
        in_dim = 3 * rel_pos_dim + scalar_dim
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_heads),
        )
        layer_ids, token_ids, block_ids = self._build_node_indices(num_layers, num_tokens, num_blocks)
        self.register_buffer("layer_ids", layer_ids, persistent=False)
        self.register_buffer("token_ids", token_ids, persistent=False)
        self.register_buffer("block_ids", block_ids, persistent=False)

    @staticmethod
    def _build_node_indices(num_layers: int, num_tokens: int, num_blocks: int):
        ids = []
        for layer in range(num_layers):
            for token in range(num_tokens):
                for block in range(num_blocks):
                    ids.append((layer, token, block))
        x = torch.tensor(ids, dtype=torch.long)
        return x[:, 0], x[:, 1], x[:, 2]

    def _relative_features(self, n: int, device: torch.device) -> torch.Tensor:
        lid = self.layer_ids[:n].to(device)
        tid = self.token_ids[:n].to(device)
        bid = self.block_ids[:n].to(device)
        dl = lid[:, None] - lid[None, :] + self.num_layers - 1
        dt = tid[:, None] - tid[None, :] + self.num_tokens - 1
        db = bid[:, None] - bid[None, :] + self.num_blocks - 1
        return torch.cat([self.layer_rel(dl), self.token_rel(dt), self.block_rel(db)], dim=-1)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        b, n, _ = nodes.shape
        feats = [self._relative_features(n, nodes.device).unsqueeze(0).expand(b, n, n, -1)]
        if self.use_cosine:
            normed = F.normalize(nodes.float(), dim=-1).to(nodes.dtype)
            cos = torch.einsum("bnd,bmd->bnm", normed, normed).unsqueeze(-1)
            feats.append(cos)
        if self.use_distance:
            dist = torch.cdist(nodes.float(), nodes.float(), p=2).to(nodes.dtype).unsqueeze(-1)
            dist = torch.log1p(dist)
            feats.append(dist)
        edge = torch.cat(feats, dim=-1)
        bias = self.edge_mlp(edge).permute(0, 3, 1, 2).contiguous()
        return bias.clamp(-self.edge_bias_clip, self.edge_bias_clip)


class RelationAwareSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_bias: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + edge_bias
        attn = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, self.model_dim)
        return self.out(out)


class RelationAwareEncoderLayer(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = RelationAwareSelfAttention(model_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, model_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * ff_mult, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), edge_bias))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x).float(), dim=1).to(x.dtype)
        return torch.sum(w * x, dim=1)
