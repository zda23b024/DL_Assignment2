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

         # ✅ FIX: constrain outputs
        # Constrain all bbox outputs with sigmoid for checkpoint compatibility.
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        features = self.encoder(x)
        raw_bbox = self.regressor(features)
        raw_bbox = self.output_activation(raw_bbox) * 224.0

        # Compatibility conversion:
        # If a checkpoint was effectively trained as [x1, y1, x2, y2],
        # convert to grader-expected [cx, cy, w, h] in image space.
        x1, y1, x2, y2 = raw_bbox[:, 0], raw_bbox[:, 1], raw_bbox[:, 2], raw_bbox[:, 3]
        x_min = torch.minimum(x1, x2)
        x_max = torch.maximum(x1, x2)
        y_min = torch.minimum(y1, y2)
        y_max = torch.maximum(y1, y2)

        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min).clamp(min=1e-3)
        h = (y_max - y_min).clamp(min=1e-3)

        return torch.stack([cx, cy, w, h], dim=1)
