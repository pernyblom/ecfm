from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class TrainConfig:
    batch_size: int
    num_steps: int
    lr: float
    device: str
    seed: int
    recon_every: int
    recon_num_patches: int
    recon_out_dir: str
    recon_upscale: int
    probe_every: int
    probe_subset_size: int
    probe_subset_seed: int
    probe_region_seed: int
    probe_regions_per_sample: int
    probe_epochs: int
    probe_batch_size: int
    probe_lr: float
    probe_cache_dir: str
    patch_loss_blur_radius: int
    count_loss_weight: float
    checkpoint_dir: str
    checkpoint_every: int
    resume_path: str
    load_model_only: bool


@dataclass
class ModelConfig:
    patch_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    decoder_embed_dim: int
    decoder_num_heads: int
    decoder_num_layers: int
    mlp_ratio: float
    plane_embed_dim: int
    metadata_dim: int
    mask_ratio: float
    use_pos_embedding: bool
    use_relative_bias: bool
    rel_bias_hidden_dim: int


@dataclass
class DataConfig:
    dataset_name: str
    dataset_path: str
    split: str
    num_samples: int
    max_samples: int
    max_events: int
    image_width: int
    image_height: int
    time_unit: float
    time_bins: int
    region_scales: list[int]
    region_scales_x: list[int]
    region_scales_y: list[int]
    region_time_scales: list[float]
    region_sampling: str
    grid_x: int
    grid_y: int
    grid_t: int
    grid_plane_mode: str
    fixed_region_sizes: bool
    plane_types: list[str]
    num_regions: int
    num_regions_choices: list[int]
    subset_seed: int
    fixed_region_seed: int
    fixed_region_positions_global: bool
    fixed_single_region: bool
    patch_divider: float
    cache_max_samples: int
    cache_token_max_samples: int
    cache_token_variants_per_sample: int
    cache_token_dir: str
    cache_token_variant_mode: str
    cache_token_clear_on_start: bool
    cache_token_config_id: str = ""
    cache_token_drop_prob: float = 0.0


@dataclass
class Config:
    train: TrainConfig
    model: ModelConfig
    data: DataConfig


def load_config(path: str | Path) -> Config:
    raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data = dict(raw["data"])
    data.setdefault("cache_token_config_id", "")
    data.setdefault("cache_token_drop_prob", 0.0)
    data.setdefault("num_regions_choices", [])
    return Config(
        train=TrainConfig(**raw["train"]),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**data),
    )


def config_hash(path: str | Path) -> str:
    cfg_path = Path(path)
    payload = f"{cfg_path.resolve()}::{cfg_path.read_text(encoding='utf-8')}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
