import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import ale_py

gym.register_envs(ale_py)


class AtariPongDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config

        self.env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

        self.stack_size = 4
        self.buffer_size = 2000

        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((config.data.image_size, config.data.image_size)),
                T.Grayscale(),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        )

        self.frames = []
        self.next_frames = []
        self.actions = []

        self._fill_buffer()

    def _fill_buffer(self):
        obs, _ = self.env.reset()

        for _ in range(self.buffer_size):
            action = self._sample_action(obs)

            self.frames.append(obs.copy())

            obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.next_frames.append(obs.copy())
            self.actions.append(action)

            if done:
                obs, _ = self.env.reset()

    def _sample_action(self, obs):
        if np.random.rand() < 0.2:
            return np.random.randint(0, 6)

        return np.random.choice([2, 3])

    def __len__(self):
        return len(self.frames) - self.stack_size - 1

    def __getitem__(self, idx):
        frames = []
        for k in range(self.stack_size):
            frames.append(self.transform(self.frames[idx + k]))

        frame_stack = torch.cat(frames, dim=0)  # (stack_size, H, W)

        next_frame = self.transform(self.next_frames[idx + self.stack_size - 1])

        action = self.actions[idx + self.stack_size - 1]

        return frame_stack, next_frame, action
