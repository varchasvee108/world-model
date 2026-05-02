from dataclasses import dataclass
from typing import Optional
import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from core.config import Config
from data.dataloader import dataloader
from model.model import NextFrameModel


@dataclass
class Components:
    device: torch.device
    tokenizer: Optional[object]
    train_loader: DataLoader
    val_loader: DataLoader
    model: NextFrameModel
    diffusion: Optional[object]
    optimizer: torch.optim.Optimizer
    scheduler: LambdaLR


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


def build_components(config: Config) -> Components:
    device = get_device()
    train_loader, val_loader = dataloader(config)

    model = NextFrameModel(config).to(device)

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.training.lr,
        betas=config.training.betas,
    )

    def lr_lambda(step):
        if step < config.training.warmup_iters:
            return step / max(1, config.training.warmup_iters)
        progress = (step - config.training.warmup_iters) / max(
            1, config.training.max_steps - config.training.warmup_iters
        )
        return 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = LambdaLR(optimizer, lr_lambda)

    return Components(
        device=device,
        tokenizer=None,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        diffusion=None,
        optimizer=optimizer,
        scheduler=scheduler,
    )
