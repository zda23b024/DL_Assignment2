"""Segmentation model
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class ConvBlock(nn.Module):
    """Simple conv block: Conv → BN → ReLU → Conv → BN → ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VGG11UNet(nn.Module):
    """U-Net style segmentation network."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11UNet, self).__init__()

        # Encoder (VGG11)
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Decoder (upsampling path)
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = ConvBlock(512 + 512, 512)

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 64)

        # Final segmentation head
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        # Encoder with skip features
        bottleneck, feats = self.encoder(x, return_features=True)

        # Skip features
        f1 = feats["block1"]  # [B, 64, 224, 224]
        f2 = feats["block2"]  # [B, 128, 112, 112]
        f3 = feats["block3"]  # [B, 256, 56, 56]
        f4 = feats["block4"]  # [B, 512, 28, 28]
        f5 = feats["block5"]  # [B, 512, 14, 14]

        # Decoder
        x = self.up5(bottleneck)              # 7 → 14
        x = torch.cat([x, f5], dim=1)
        x = self.dec5(x)

        x = self.up4(x)                       # 14 → 28
        x = torch.cat([x, f4], dim=1)
        x = self.dec4(x)

        x = self.up3(x)                       # 28 → 56
        x = torch.cat([x, f3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)                       # 56 → 112
        x = torch.cat([x, f2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)                       # 112 → 224
        x = torch.cat([x, f1], dim=1)
        x = self.dec1(x)

        # Final output
        out = self.final(x)  # [B, num_classes, 224, 224]

        return out