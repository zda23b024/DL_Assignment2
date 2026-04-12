import torch
import torch.nn as nn
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

        # ✅ FIX: constrain outputs
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        features = self.encoder(x)
        bbox = self.regressor(features)

        # ✅ FIX: scale to image space (224x224)
        bbox = self.output_activation(bbox) * 224.0

        return bbox
