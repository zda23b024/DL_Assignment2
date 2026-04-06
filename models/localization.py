"""Localization modules
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer.

    This model reuses the VGG11 convolutional backbone as the feature extractor
    and fine-tunes it for bounding-box regression. Fine-tuning is chosen here
    because the pretrained classification encoder provides strong spatial
    features, while localization still benefits from adapting those features
    to predict precise object coordinates.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        freeze_encoder: bool = False,
    ):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
            freeze_encoder: If True, freeze the VGG11 encoder weights.
        """
        super(VGG11Localizer, self).__init__()

        # Shared encoder (same as classification)
        self.encoder = VGG11Encoder(in_channels=in_channels)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),  # [B, 512, 7, 7] → [B, 25088]

            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(512, 4)  # Output: [x_center, y_center, width, height]
        )

        self.output_activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4]
        """
        # Extract deep features
        features = self.encoder(x)  # [B, 512, 7, 7]

        # Predict bounding boxes
        bbox = self.regressor(features)  # [B, 4]

        bbox_xy = bbox[:, :2]
        bbox_wh = self.output_activation(bbox[:, 2:]) + 1e-3
        bbox = torch.cat([bbox_xy, bbox_wh], dim=1)

        return bbox