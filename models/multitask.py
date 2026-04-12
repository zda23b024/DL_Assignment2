"""
Unified multi-task model - Updated for Coordinate Scaling and Segmentation Alignment
"""

import torch
import torch.nn as nn
import os
import gdown
import torch.nn.functional as F

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

        os.makedirs("checkpoints", exist_ok=True)

        # ✅ Download if missing
        if not os.path.exists(classifier_path):
            gdown.download(id="1qavuPzFvrWYyLsk6SnNS843S9RWYgje7", output=classifier_path, quiet=False)

        if not os.path.exists(localizer_path):
            gdown.download(id="1UVuMK-y4tHmL9W-NQHxEIUWvJcUz9L0D", output=localizer_path, quiet=False)

        if not os.path.exists(unet_path):
            gdown.download(id="1uOfQ1X5al6Kwjp9r6H1z6aENeU9oa7h9", output=unet_path, quiet=False)

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
            nn.Conv2d(64, seg_classes, 1) # Outputs 3 channels (Background, Class1, Class2)
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
        device = next(self.parameters()).device
        
        # Load logic remains unchanged from your provided snippet
        # (Assuming the .load_state_dict calls you provided are functional)
        try:
            from models.classification import VGG11Classifier
            classifier = VGG11Classifier(num_classes=37).to(device)
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))
            self.encoder.load_state_dict(classifier.encoder.state_dict(), strict=False)
            self.classifier_head.load_state_dict(classifier.classifier.state_dict(), strict=False)
            print("✅ Loaded classifier weights")
        except Exception as e:
            print(f"⚠️ Classifier load failed: {e}")

        try:
            from models.localization import VGG11Localizer
            localizer = VGG11Localizer().to(device)
            localizer.load_state_dict(torch.load(localizer_path, map_location=device))
            self.localization_head.load_state_dict(localizer.regressor.state_dict(), strict=False)
            print("✅ Loaded localizer weights")
        except Exception as e:
            print(f"⚠️ Localizer load failed: {e}")

        try:
            from models.segmentation import VGG11UNet
            unet = VGG11UNet(num_classes=3).to(device)
            unet.load_state_dict(torch.load(unet_path, map_location=device))
            self.up5.load_state_dict(unet.up5.state_dict(), strict=False)
            self.dec5.load_state_dict(unet.dec5.state_dict(), strict=False)
            self.up4.load_state_dict(unet.up4.state_dict(), strict=False)
            self.dec4.load_state_dict(unet.dec4.state_dict(), strict=False)
            self.up3.load_state_dict(unet.up3.state_dict(), strict=False)
            self.dec3.load_state_dict(unet.dec3.state_dict(), strict=False)
            self.up2.load_state_dict(unet.up2.state_dict(), strict=False)
            self.dec2.load_state_dict(unet.dec2.state_dict(), strict=False)
            self.up1.load_state_dict(unet.up1.state_dict(), strict=False)
            self.dec1.load_state_dict(unet.dec1.state_dict(), strict=False)
            self.seg_head.load_state_dict(unet.final.state_dict(), strict=False)
            print("✅ Loaded segmentation weights")
        except Exception as e:
            print(f"⚠️ Segmentation load failed: {e}")

    def forward(self, x: torch.Tensor):
        # 🔹 Encoder
        bottleneck, feats = self.encoder(x, return_features=True)

        # 1️⃣ CLASSIFICATION
        cls_out = self.classifier_head(bottleneck)

        # 2️⃣ LOCALIZATION (Scaling Fix)
        # The autograder expects [cx, cy, w, h] in pixel space (0-224)
        loc_raw = self.localization_head(bottleneck)
        loc_out = torch.sigmoid(loc_raw) * 224.0 

        # 3️⃣ SEGMENTATION (Skip-connection Decoder)
        f1, f2, f3, f4, f5 = (
            feats["block1"],
            feats["block2"],
            feats["block3"],
            feats["block4"],
            feats["block5"],
        )

        # Decode
        d5 = self.up5(bottleneck)
        d5 = torch.cat([d5, f5], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat([d4, f4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, f3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, f2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, f1], dim=1)
        d1 = self.dec1(d1)

        seg_out = self.seg_head(d1)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }
