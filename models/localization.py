import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    def __init__(self, in_channels=3, dropout_p=0.5, freeze_encoder=False):
        super(VGG11Localizer, self).__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(512, 4)
        )

        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        features = self.encoder(x)
        raw_bbox = self.regressor(features)

        # Centers in image space
        center = self.output_activation(raw_bbox[:, :2]) * 224.0
        # Positive width/height with smoother gradients near extremes
        size = F.softplus(raw_bbox[:, 2:])
        size = torch.clamp(size, min=1e-3, max=224.0)

        return torch.cat([center, size], dim=1)
