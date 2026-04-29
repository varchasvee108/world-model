from datasets import load_dataset
from torch.utils.data import Dataset
import torchvision.transforms as T
from core.config import Config


class HeistDataset(Dataset):
    def __init__(self, config: Config, split: str):
        ds = load_dataset(config.data.dataset, name="heist", split=split)
        self.frames = ds["image"]
        self.actions = ds["action"]
        self.tf = T.Compose(
            [
                T.Resize(config.data.image_size),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.frames) - 1

    def __getitem__(self, i):
        frame_t = self.tf(self.frames[i])
        frame_t_plus_1 = self.tf(self.frames[i + 1])
        action = self.actions[i]

        return frame_t, frame_t_plus_1, action
