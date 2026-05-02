import torch
import torch.nn as nn
from core.config import Config
from model.layers import ResBlock, Downsample, Upsample


class NextFrameModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.num_actions = 6

        in_ch = config.data.in_channels
        out_ch = config.data.out_channels

        self.action_emb = nn.Embedding(self.num_actions, config.model.n_embd)

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )

        self.down1 = nn.Sequential(
            ResBlock(64, 128, config.model.n_embd, dropout=config.model.dropout),
            Downsample(128),
        )

        self.down2 = nn.Sequential(
            ResBlock(128, 256, config.model.n_embd, dropout=config.model.dropout),
            Downsample(256),
        )

        self.bottleneck = ResBlock(
            256, 256, config.model.n_embd, dropout=config.model.dropout
        )

        self.up1 = nn.Sequential(
            Upsample(256),
            ResBlock(512, 128, config.model.n_embd, dropout=config.model.dropout),
        )

        self.up2 = nn.Sequential(
            Upsample(128),
            ResBlock(256, 64, config.model.n_embd, dropout=config.model.dropout),
        )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_ch, 3, padding=1),
        )

    def forward(self, frame: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        emb = self.action_emb(action)

        x = self.conv_in(frame)

        x = self.down1[0](x, emb)
        skip1 = x
        x = self.down1[1](x)

        x = self.down2[0](x, emb)
        skip2 = x
        x = self.down2[1](x)

        x = self.bottleneck(x, emb)

        x = self.up1[0](x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up1[1](x, emb)

        x = self.up2[0](x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up2[1](x, emb)

        out = self.conv_out(x)
        return out
