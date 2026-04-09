"""
Unified multi-task model
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
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
        super(MultiTaskPerceptionModel, self).__init__()

        os.makedirs("checkpoints", exist_ok=True)

        # ✅ Download if missing
        if not os.path.exists(classifier_path):
            gdown.download(id="1mO8NGNLbwF7ON59EmDUJootHJQquDIjY", output=classifier_path, quiet=False)

        if not os.path.exists(localizer_path):
            gdown.download(id="1gLofWzBacZUEAQxwIeYoZCp_cwEtz1sE", output=localizer_path, quiet=False)

        if not os.path.exists(unet_path):
            gdown.download(id="17mtEgWAkSpf13zqk1hKNDnw8WfKPA_C_", output=unet_path, quiet=False)

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

        # 🔹 Segmentation Decoder (UNet style)
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
        device = next(self.parameters()).device

        # ✅ Load classifier
        try:
            from models.classification import VGG11Classifier
            classifier = VGG11Classifier(num_classes=37).to(device)
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))

            self.encoder.load_state_dict(classifier.encoder.state_dict(), strict=False)
            self.classifier_head.load_state_dict(classifier.classifier.state_dict(), strict=False)

            print("✅ Loaded classifier weights")
        except Exception as e:
            print(f"⚠️ Classifier load failed: {e}")

        # ✅ Load localizer
        try:
            from models.localization import VGG11Localizer
            localizer = VGG11Localizer().to(device)
            localizer.load_state_dict(torch.load(localizer_path, map_location=device))

            self.localization_head.load_state_dict(localizer.regressor.state_dict(), strict=False)

            print("✅ Loaded localizer weights")
        except Exception as e:
            print(f"⚠️ Localizer load failed: {e}")

        # ✅ Load segmentation
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

        # 🔹 Classification
        cls_out = self.classifier_head(bottleneck)

        # 🔹 Localization (🔥 FIXED)
        loc_raw = self.localization_head(bottleneck)

        loc_xy = torch.sigmoid(loc_raw[:, :2])  # center in [0,1]
        loc_wh = torch.nn.functional.softplus(loc_raw[:, 2:]) + 1e-3  # positive size

        loc_out = torch.cat([loc_xy, loc_wh], dim=1)

        # 🔹 Segmentation
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
