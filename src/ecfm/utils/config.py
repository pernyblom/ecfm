from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class ModelConfig:
    patch_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_ratio: float
    plane_embed_dim: int
    metadata_dim: int
    mask_ratio: float


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
    time_bins: int
    region_scales: list[int]
    plane_types: list[str]
    num_regions: int


@dataclass
class Config:
    train: TrainConfig
    model: ModelConfig
    data: DataConfig


def load_config(path: str | Path) -> Config:
    raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(
        train=TrainConfig(**raw["train"]),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
    )
