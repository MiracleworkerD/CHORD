from __future__ import annotations

import torch
import torch.nn as nn

from .config import DataConfig, ModelConfig
from .layers import AttentionPooling, EdgeRelationBias, RelationAwareEncoderLayer, ScatteredPadder

class SharedBlockProjector(nn.Module):
    def __init__(self, chunk_size: int, model_dim: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(chunk_size),
            nn.Linear(chunk_size, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        return x + self.ff(x)


class RelationalHallucinationDetector(nn.Module):
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig):
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        if model_cfg.max_dim % model_cfg.num_blocks != 0:
            raise ValueError("max_dim must be divisible by num_blocks")
        self.num_tokens = data_cfg.last_k_tokens
        self.num_layers = data_cfg.last_l_layers
        self.num_blocks = model_cfg.num_blocks
        self.chunk_size = model_cfg.max_dim // model_cfg.num_blocks
        self.padder = ScatteredPadder(model_cfg.max_dim, model_cfg.use_scattered_padding)
        # self.block_projector = nn.Linear(self.chunk_size, model_cfg.model_dim)
        self.block_projector = SharedBlockProjector(
            chunk_size=self.chunk_size,
            model_dim=model_cfg.model_dim,
            dropout=model_cfg.dropout,
        )
        self.layer_embed = nn.Embedding(model_cfg.layer_embed_size, model_cfg.model_dim)
        self.token_embed = nn.Embedding(model_cfg.token_embed_size, model_cfg.model_dim)
        self.block_embed = nn.Embedding(model_cfg.block_embed_size, model_cfg.model_dim)
        self.edge_bias = EdgeRelationBias(
            num_layers=self.num_layers,
            num_tokens=self.num_tokens,
            num_blocks=self.num_blocks,
            rel_pos_dim=model_cfg.rel_pos_dim,
            num_heads=model_cfg.num_heads,
            dropout=model_cfg.dropout,
            use_cosine=model_cfg.use_content_cosine,
            use_distance=model_cfg.use_content_distance,
            edge_bias_clip=model_cfg.edge_bias_clip,
        )
        self.encoder = nn.ModuleList([
            RelationAwareEncoderLayer(
                model_dim=model_cfg.model_dim,
                num_heads=model_cfg.num_heads,
                ff_mult=model_cfg.ff_mult,
                dropout=model_cfg.dropout,
            )
            for _ in range(model_cfg.num_encoder_layers)
        ])
        self.pool = AttentionPooling(model_cfg.model_dim, model_cfg.dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_cfg.model_dim),
            nn.Linear(model_cfg.model_dim, model_cfg.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(model_cfg.classifier_dropout),
            nn.Linear(model_cfg.classifier_hidden_dim, 1),
        )
        layer_ids, token_ids, block_ids = self._build_node_indices()
        self.register_buffer("layer_ids", layer_ids, persistent=False)
        self.register_buffer("token_ids", token_ids, persistent=False)
        self.register_buffer("block_ids", block_ids, persistent=False)

    def _build_node_indices(self):
        ids = []
        for layer in range(self.num_layers):
            for token in range(self.num_tokens):
                for block in range(self.num_blocks):
                    ids.append((layer, token, block))
        x = torch.tensor(ids, dtype=torch.long)
        return x[:, 0], x[:, 1], x[:, 2]

    def encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, tokens, layers, hidden_dim]")
        x = self.padder(x)
        b, t, l, _ = x.shape
        if t != self.num_tokens or l != self.num_layers:
            raise ValueError(f"expected tokens/layers {(self.num_tokens, self.num_layers)}, got {(t, l)}")
        x = x.view(b, t, l, self.num_blocks, self.chunk_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        nodes = self.block_projector(x).view(b, self.num_layers * self.num_tokens * self.num_blocks, -1)
        n = nodes.shape[1]
        nodes = (
            nodes
            + self.layer_embed(self.layer_ids[:n]).unsqueeze(0)
            + self.token_embed(self.token_ids[:n]).unsqueeze(0)
            + self.block_embed(self.block_ids[:n]).unsqueeze(0)
        )
        return nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes = self.encode_nodes(x)
        edge_bias = self.edge_bias(nodes)
        for layer in self.encoder:
            nodes = layer(nodes, edge_bias)
        z = self.pool(nodes)
        return self.classifier(z)
