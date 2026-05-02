from pathlib import Path
import tomllib
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    seed: int = Field(gt=0)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    batch_size: int = Field(gt=0)
    image_size: int = Field(gt=0)
    in_channels: int = Field(gt=0)
    out_channels: int = Field(gt=0)
    num_workers: int = Field(gt=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_embd: int
    hidden_dim: int
    n_head: int
    num_layers: int
    dropout: float
    timestep_n_embd: int


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: float = Field(gt=0)
    max_steps: int = Field(gt=0)
    betas: tuple[float, float] = Field(ge=0, le=1)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(gt=0)
    warmup_iters: int = Field(ge=0)
    eval_interval: int
    log_interval: int
    grad_accum_steps: int
    scheduler: Literal["cosine", "cosine_with_restarts", "linear", "constant"] = (
        "cosine_with_restarts"
    )


class DiffusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timesteps: int = Field(gt=0)
    noise_schedule: str
    beta_start: float = Field(gt=0)
    beta_end: float = Field(gt=0)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    diffusion: DiffusionConfig

    @classmethod
    def load_config(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "rb") as f:
            toml_dict = tomllib.load(f)
        return cls.model_validate(toml_dict)
