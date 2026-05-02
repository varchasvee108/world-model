# scripts/train.py
from core.config import Config
from core.factory import build_components
from Trainer.trainer import Trainer


def main():
    config = Config.load_config("config/config.toml")
    comp = build_components(config)

    trainer = Trainer(
        config=config,
        model=comp.model,
        optimizer=comp.optimizer,
        scheduler=comp.scheduler,
        train_dataloader=comp.train_loader,
        val_dataloader=comp.val_loader,
        device=comp.device,
        tokenizer=comp.tokenizer,
        diffusion=comp.diffusion,
    )
    trainer.train()


if __name__ == "__main__":
    main()
