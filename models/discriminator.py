import torch
import torch.nn as nn

from models.blocks import ConvBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super(Discriminator, self).__init__()
        dim = 32
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            ConvBlock(dim * 1, dim * 2, 4, 2, 1),
            ConvBlock(dim * 2, dim * 4, 4, 2, 1),
            ConvBlock(dim * 4, dim * 8, 4, 2, 1),
            nn.Conv2d(dim * 8, dim * 8, 4, 2, 1, bias=False),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16384, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.head(x)
        return x
