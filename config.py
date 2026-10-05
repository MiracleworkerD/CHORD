from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class DataConfig:
    source_llm_dirs: Dict[str, str] = field(default_factory=lambda: {
        "qwen3_14b": "./experiments/Qwen/Qwen3-14B/halueval_qa_5000samples",
        "llm_b": "./experiments/LLM_B/halueval_qa_5000samples",
        "llm_c": "./experiments/LLM_C/halueval_qa_5000samples",
    })
    target_llm_dirs: Dict[str, str] = field(default_factory=dict)
    hidden_filename: str = "02_hidden_states.h5"
    metadata_filename: str = "03_final_scored_metadata.jsonl"
    token_group_name: str = "generated_tokens"
    layer_prefix: str = "layer_"
    sample_id_key: str = "sample_id"
    label_key: str = "eval_category"
    positive_label: str = "hallucination"
    negative_labels: Tuple[str, ...] = ("correct", "abstention")
    last_k_tokens: int = 2
    last_l_layers: int = 3
    val_ratio: float = 0.2
    num_workers: int = 0
    pin_memory: bool = True
    cache_in_memory: bool = True


@dataclass
class ModelConfig:
    max_dim: int = 8192
    num_blocks: int = 64
    model_dim: int = 256
    rel_pos_dim: int = 48
    num_heads: int = 8
    num_encoder_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.1
    classifier_hidden_dim: int = 128
    classifier_dropout: float = 0.3
    layer_embed_size: int = 64
    token_embed_size: int = 64
    block_embed_size: int = 128
    use_scattered_padding: bool = True
    use_content_distance: bool = True
    use_content_cosine: bool = True
    edge_bias_clip: float = 5.0


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    precision: str = "bf16"
    batch_size: int = 32
    eval_batch_size: int = 128
    episodes: int = 3000
    inner_steps: int = 2
    min_meta_train_domains: int = 2
    max_meta_train_domains: int = 4
    inner_lr: float = 5e-4
    meta_lr: float = 0.1
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    grad_clip_norm: float = 1.0
    eval_every: int = 50
    save_every: int = 200
    output_dir: str = "./outputs/relattn_hallu"
    checkpoint_name: str = "best.pt"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def prepare(self) -> "Config":
        Path(self.train.output_dir).mkdir(parents=True, exist_ok=True)
        return self


cfg = Config().prepare()
