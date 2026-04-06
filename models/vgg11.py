"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from models.layers import CustomDropout


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super(VGG11Encoder, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass."""

        features = {}

        # Block 1
        x = self.block1(x)
        features["block1"] = x
        x = self.pool1(x)

        # Block 2
        x = self.block2(x)
        features["block2"] = x
        x = self.pool2(x)

        # Block 3
        x = self.block3(x)
        features["block3"] = x
        x = self.pool3(x)

        # Block 4
        x = self.block4(x)
        features["block4"] = x
        x = self.pool4(x)

        # Block 5
        x = self.block5(x)
        features["block5"] = x
        x = self.pool5(x)

        # Final bottleneck
        bottleneck = x

        if return_features:
            return bottleneck, features

        return bottleneck


class VGG11(nn.Module):
    """VGG11 classification model with BatchNorm and custom Dropout."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11, self).__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)
        return x
