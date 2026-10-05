"""Compact DS-CNN for binary keyword spotting.

Input:  (batch, 1, n_mels=40, n_frames=98)  -- matches features.py output
Output: (batch, 1) raw logit. Apply sigmoid at inference time.

Why depthwise-separable convolutions?
    A standard KxK conv with C_in -> C_out costs K*K*C_in*C_out weights.
    A depthwise-separable conv splits it into:
        depthwise:  K*K*C_in         (one KxK filter per input channel)
        pointwise:  1*1*C_in*C_out   (mixes channels)
    This cuts parameters and multiply-accumulates by roughly K*K times,
    which is why DS-CNNs are the de-facto standard for KWS on
    microcontrollers / DSPs (see ARM's "Hello Edge" paper).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DSConvBlock(nn.Module):
    """Depthwise 3x3 conv -> BN -> ReLU -> pointwise 1x1 conv -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
            groups=in_ch, bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_dw(self.depthwise(x)))
        x = self.relu(self.bn_pw(self.pointwise(x)))
        return x


class DSCNN(nn.Module):
    """Small DS-CNN: stem conv -> 4 DS blocks -> global average pool -> FC.

    ~20k parameters with default settings: small enough to quantize and
    deploy on an embedded DSP later (Chunk 3 territory).
    """

    def __init__(self, n_channels: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        c = n_channels
        # Stem: full conv, downsamples both mel and time axes by 2.
        # (1, 40, 98) -> (c, 20, 49)
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DSConvBlock(c, c),
            DSConvBlock(c, c),
            DSConvBlock(c, c),
            DSConvBlock(c, c),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # (c, H, W) -> (c, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)   # (batch, c)
        x = self.dropout(x)
        return self.fc(x)             # (batch, 1) logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = DSCNN()
    x = torch.randn(2, 1, 40, 98)
    y = m(x)
    print(f"output shape: {tuple(y.shape)}")          # (2, 1)
    print(f"parameters:   {count_parameters(m):,}")
