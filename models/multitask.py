"""Unified multi-task model
"""

import torch
import torch.nn as nn
import os
import gdown

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
    ):
        super(MultiTaskPerceptionModel, self).__init__()

        # 🔥 Ensure checkpoints folder exists
        os.makedirs("checkpoints", exist_ok=True)

        classifier_path = "checkpoints/classifier.pth"
        localizer_path = "checkpoints/localizer.pth"
        unet_path = "checkpoints/unet.pth"

        # 🔥 Download models if not present
        if not os.path.exists(classifier_path):
            gdown.download(id="1eAj0XNOQsJYp89HoQZzc_oIAgoJW4q6v", output=classifier_path, quiet=False)

        if not os.path.exists(localizer_path):
            gdown.download(id="1UG9OKG6YRlfjuTqWIZZl_SZkBOGjBV2i", output=localizer_path, quiet=False)

        if not os.path.exists(unet_path):
            gdown.download(id="1MhM1BoWxFEW2TBblClRw0M9V5g_PrffO", output=unet_path, quiet=False)

        # 🔹 Shared Encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # 🔹 Classification Head
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
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

        # 🔹 Segmentation Decoder
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

        # 🔥 Load weights safely
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
        device = torch.device("cpu")

        try:
            self.encoder.load_state_dict(
                torch.load(classifier_path, map_location=device),
                strict=False
            )
            print("✅ Loaded encoder from classifier")
        except Exception as e:
            print(f"⚠️ Could not load classifier weights: {e}")

        # (Optional) load segmentation if compatible
        try:
            seg_weights = torch.load(unet_path, map_location=device)
            self.load_state_dict(seg_weights, strict=False)
            print("✅ Loaded segmentation weights")
        except Exception as e:
            print(f"⚠️ Segmentation weights skipped: {e}")

    def forward(self, x: torch.Tensor):
        bottleneck, feats = self.encoder(x, return_features=True)

        cls_out = self.classifier_head(bottleneck)

        loc_out = self.localization_head(bottleneck)
        loc_out_xy = loc_out[:, :2]
        loc_out_wh = torch.nn.functional.softplus(loc_out[:, 2:]) + 1e-3
        loc_out = torch.cat([loc_out_xy, loc_out_wh], dim=1)

        f1, f2, f3, f4, f5 = (
            feats["block1"],
            feats["block2"],
            feats["block3"],
            feats["block4"],
            feats["block5"],
        )

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
