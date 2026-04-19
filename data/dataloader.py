from torch.utils.data import DataLoader
from core.config import Config
from data.heist_dataset import HeistDataset


def dataloader(config: Config):
    train_ds = HeistDataset(config=config, split="train")
    val_ds = HeistDataset(config=config, split="validation")

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.data.num_workers,
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        pin_memory=True,
        num_workers=config.data.num_workers,
        persistent_workers=True,
    )

    return train_dataloader, val_dataloader
