import torch
import torch.nn as nn

Normalization = nn.BatchNorm2d
Activation = nn.ReLU


class ConvBlock(nn.Module):
    def __init__(
        self, inp: int, outp: int, k: int = 3, s: int = 1, p: int = 0
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inp, outp, k, s, p, bias=False)
        self.norm = Normalization(outp)
        self.act = Activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResidualConv(nn.Module):
    def __init__(self, inp: int, outp: int, s: int = 1, p: int = 0):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            Normalization(inp),
            Activation(inplace=True),
            nn.Conv2d(
                inp, outp, kernel_size=3, stride=s, padding=p, bias=False
            ),
            Normalization(outp),
            Activation(inplace=True),
            nn.Conv2d(outp, outp, kernel_size=3, padding=1, bias=False),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(
                inp, outp, kernel_size=3, stride=s, padding=1, bias=False
            ),
            Normalization(outp),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) + self.conv_skip(x)
