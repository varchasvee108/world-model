from torch.utils.data import DataLoader, random_split
from core.config import Config
from data.atari_dataset import AtariPongDataset


def dataloader(config: Config):
    full_ds = AtariPongDataset(config=config)

    val_size = int(0.1 * len(full_ds))
    train_size = len(full_ds) - val_size

    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.data.num_workers,
        persistent_workers=config.data.num_workers > 0,
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.data.num_workers,
        persistent_workers=config.data.num_workers > 0,
    )

    return train_dataloader, val_dataloader
