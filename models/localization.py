"""
Localization modules - Updated for Pixel-Space Scaling
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout

class VGG11Localizer(nn.Module):
    """
    VGG11-based localizer updated to output [cx, cy, w, h] in the 
    [0, 224] image space required by the autograder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        freeze_encoder: bool = False,
    ):
        super(VGG11Localizer, self).__init__()

        # 🔹 Shared encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # 🔹 Regression head
        # Standard VGG11 features are 512x7x7 = 25088
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(512, 4)  # Output: [cx, cy, w, h]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns coordinates scaled to 224.0 pixels.
        """
        # Extract features
        features = self.encoder(x)

        # 1. Get raw predictions
        bbox = self.regressor(features)

        # 2. Map outputs to [0, 1] then scale to [0, 224]
        # This addresses the "Coordinates are not in image space" error.
        bbox = torch.sigmoid(bbox) * 224.0

        return bbox
