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

        # 1) Regress 4 values in [0, 224] image space.
        # We parameterize these as corner points and then convert to
        # [cx, cy, w, h] so the public model contract is always respected.
        corners = torch.sigmoid(self.regressor(features)) * 224.0
        x1, y1, x2, y2 = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]

        # 2) Make corner ordering robust even if the checkpoint was trained
        # with swapped endpoints.
        left = torch.minimum(x1, x2)
        right = torch.maximum(x1, x2)
        top = torch.minimum(y1, y2)
        bottom = torch.maximum(y1, y2)

        # 3) Convert to [cx, cy, w, h] in image-space pixels.
        cx = (left + right) * 0.5
        cy = (top + bottom) * 0.5
        w = (right - left).clamp(min=1e-6)
        h = (bottom - top).clamp(min=1e-6)

        return torch.stack([cx, cy, w, h], dim=1)
