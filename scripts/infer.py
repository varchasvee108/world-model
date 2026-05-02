import torch
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
    )

    ckpt = torch.load("checkpoints/final.pt", map_location=comp.device)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.model.eval()

    frame_t, _, action = next(iter(comp.val_loader))
    frame_t = frame_t.to(comp.device)
    action = action.to(comp.device)

    with torch.no_grad():
        pred = trainer.model(frame_t, action)

    print(pred.shape)


if __name__ == "__main__":
    main()
