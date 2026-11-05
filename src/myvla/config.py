from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    backbone: str = "openai/clip-vit-base-patch16"
    revision: str = "main"
    backend: str = "clip"
    image_size: int = 224
    hidden_dim: int = 256
    layers: int = 4
    heads: int = 8
    chunk_size: int = 8
    dropout: float = 0.1


@dataclass
class DataConfig:
    directory: str = "data/libero_spatial"
    pattern: str = "*_demo.hdf5"
    val_fraction: float = 0.1
    flip_vertical: bool = True


@dataclass
class TrainConfig:
    output_dir: str = "runs/spatial_v1"
    device: str = "cuda"
    precision: str = "fp16"
    batch_size: int = 2
    grad_accumulation: int = 8
    workers: int = 2
    steps: int = 30000
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    validate_every: int = 500
    validation_batches: int = 100
    save_every: int = 1000
    log_every: int = 20
    seed: int = 42
    image_shift: int = 0
    color_jitter: float = 0.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self):
        m, d, t = self.model, self.data, self.train
        if m.backend not in {"clip", "tiny"}:
            raise ValueError("model.backend must be clip or tiny (offline smoke test only)")
        if min(m.hidden_dim, m.layers, m.heads, m.chunk_size, m.image_size) < 1:
            raise ValueError("Model dimensions must be positive")
        if m.hidden_dim % m.heads or not 0 <= m.dropout < 1:
            raise ValueError("Invalid attention dimensions or dropout")
        if m.backend == "clip" and m.image_size != 224:
            raise ValueError("This CLIP baseline requires image_size=224")
        if m.backend == "tiny" and m.image_size % 4:
            raise ValueError("Tiny CLIP image_size must be divisible by 4")
        if not 0 < d.val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        if (
            min(
                t.batch_size,
                t.grad_accumulation,
                t.steps,
                t.validate_every,
                t.validation_batches,
                t.save_every,
                t.log_every,
            )
            < 1
            or t.workers < 0
        ):
            raise ValueError("Invalid training counts")
        if t.lr <= 0 or t.weight_decay < 0 or t.warmup_steps < 0 or t.grad_clip <= 0:
            raise ValueError("Invalid optimizer configuration")
        if t.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if not isinstance(t.image_shift, int) or not 0 <= t.image_shift < m.image_size:
            raise ValueError("image_shift must be an integer in [0, image_size)")
        if not 0 <= t.color_jitter < 1:
            raise ValueError("color_jitter must be in [0, 1)")
        return self

    def to_dict(self):
        return asdict(self)


def from_dict(raw):
    unknown = set(raw) - {"model", "data", "train"}
    if unknown:
        raise ValueError(f"Unknown config sections: {unknown}")
    return Config(
        ModelConfig(**raw.get("model", {})),
        DataConfig(**raw.get("data", {})),
        TrainConfig(**raw.get("train", {})),
    ).validate()


def load_config(path):
    return from_dict(yaml.safe_load(Path(path).read_text()) or {})
