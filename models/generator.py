import torch
import torch.nn as nn

from models.blocks import ConvBlock


class UnetGenerator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dim=16):
        super().__init__()
        self.e0 = nn.Sequential(
            ConvBlock(in_channels, dim * 1, 3, 1, 1),
            ConvBlock(dim * 1, dim * 1, 3, 1, 1),
            # OUTPUT 256x256x16
        )

        self.e1 = nn.Sequential(
            ConvBlock(dim * 1, dim * 2, 3, 1, 1),
            ConvBlock(dim * 2, dim * 2, 4, 2, 1),
            # OUTPUT 128x128x16
        )

        self.e2 = nn.Sequential(
            ConvBlock(dim * 2, dim * 4, 3, 1, 1),
            ConvBlock(dim * 4, dim * 4, 4, 2, 1),
            # OUTPUT 64x64x32
        )

        self.e3 = nn.Sequential(
            ConvBlock(dim * 4, dim * 8, 3, 1, 1),
            ConvBlock(dim * 8, dim * 8, 4, 2, 1),
            # OUTPUT 32x32x64
        )

        self.e4 = nn.Sequential(
            ConvBlock(dim * 8, dim * 16, 3, 1, 1),
            ConvBlock(dim * 16, dim * 16, 4, 2, 1),
            # OUTPUT 16x16x128
        )

        self.middle = ConvBlock(dim * 16, dim * 128, 4, 2, 1)
        self.style = nn.Sequential(
            nn.Linear(4096, dim * 128),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.d4_0 = nn.Sequential(
            self.upsample,
            ConvBlock(dim * 128, dim * 32, 3, 1, 1),
        )
        self.d4_1 = nn.Sequential(
            ConvBlock(dim * 32, dim * 32, 3, 1, 1),
        )

        self.d3 = nn.Sequential(
            self.upsample,
            ConvBlock(dim * 32 + dim * 16, dim * 8, 3, 1, 1),
            ConvBlock(dim * 8, dim * 8, 3, 1, 1),
        )
        self.d2 = nn.Sequential(
            self.upsample,
            ConvBlock(dim * 8 + dim * 8, dim * 4, 3, 1, 1),
            ConvBlock(dim * 4, dim * 4, 3, 1, 1),
        )

        self.d1 = nn.Sequential(
            self.upsample,
            ConvBlock(dim * 4 + dim * 4, dim * 2, 3, 1, 1),
            ConvBlock(dim * 2, dim * 2, 3, 1, 1),
        )
        self.last = nn.Sequential(
            self.upsample,
            ConvBlock(dim * 2 + dim * 2, dim * 2, 3, 1, 1),
            nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, f: torch.Tensor):
        ##### Encoder ######
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        ##### Middle  ######
        m = self.middle(e4)
        f = self.style(f).view([-1, f.shape[1] // 2, 1, 1])
        x = m + f
        ##### Decoder ######
        d4 = self.d4_0(x)
        x = torch.cat([self.d4_1(d4), e4], dim=1)
        x = torch.cat([self.d3(x), e3], dim=1)
        x = torch.cat([self.d2(x), e2], dim=1)
        x = torch.cat([self.d1(x), e1], dim=1)
        x = self.last(x)
        return x, e4, d4

    def predict(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        x, _, _ = self(x, f)
        return x
