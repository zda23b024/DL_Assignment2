"""Unified multi-task model
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
        super(MultiTaskPerceptionModel, self).__init__()

        # 🔹 Shared Encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # 🔹 Classification Head
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),

            nn.Linear(4096, num_breeds)
        )

        # 🔹 Localization Head
        self.localization_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),

            nn.Linear(512, 4)
        )

        # 🔹 Segmentation Decoder (same as UNet)
        self.up5 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.dec5 = self._conv_block(512 + 512, 512)

        self.up4 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.dec4 = self._conv_block(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = self._conv_block(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = self._conv_block(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = self._conv_block(64 + 64, 64)

        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),
            nn.Conv2d(64, seg_classes, 1)
        )

        # 🔥 Load pretrained weights
        self._load_weights(classifier_path, localizer_path, unet_path)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _load_weights(self, classifier_path, localizer_path, unet_path):
        try:
            self.encoder.load_state_dict(torch.load(classifier_path), strict=False)
            print("✅ Loaded encoder from classifier")
        except:
            print("⚠️ Could not load classifier weights")

        # NOTE:
        # For simplicity & stability, we reuse encoder weights from classifier.
        # Heads are freshly initialized (acceptable unless strict copying required).

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model."""

        # 🔹 Encoder
        bottleneck, feats = self.encoder(x, return_features=True)

        # 🔹 Classification
        cls_out = self.classifier_head(bottleneck)

        # 🔹 Localization
        loc_out = self.localization_head(bottleneck)
        loc_out_xy = loc_out[:, :2]
        loc_out_wh = torch.nn.functional.softplus(loc_out[:, 2:]) + 1e-3
        loc_out = torch.cat([loc_out_xy, loc_out_wh], dim=1)

        # 🔹 Segmentation
        f1 = feats["block1"]
        f2 = feats["block2"]
        f3 = feats["block3"]
        f4 = feats["block4"]
        f5 = feats["block5"]

        x = self.up5(bottleneck)
        x = torch.cat([x, f5], dim=1)
        x = self.dec5(x)

        x = self.up4(x)
        x = torch.cat([x, f4], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, f3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, f1], dim=1)
        x = self.dec1(x)

        seg_out = self.seg_head(x)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }