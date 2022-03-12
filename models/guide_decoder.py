from tkinter import N
import torch
import torch.nn as nn
from models.blocks import ConvBlock


class GuideDecoder(nn.Module):
    def __init__(self, inp: int, outp: int, dim=512) -> None:
        super().__init__()
        upsample = nn.Upsample(scale_factor=2)
        self.block = nn.Sequential(
            ConvBlock(inp, dim // 1, 3, 1, 1),
            upsample,  # 32x32x256
            ConvBlock(dim // 1, dim // 2, 3, 1, 1),
            upsample,  # 64x64x128
            ConvBlock(dim // 2, dim // 4, 3, 1, 1),
            upsample,  # 128x128x64
            ConvBlock(dim // 4, dim // 8, 3, 1, 1),
            upsample,  # 256x256x3
            nn.Conv2d(dim // 8, outp, 1, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x
