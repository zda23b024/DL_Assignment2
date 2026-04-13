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
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


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
            gdown.download(id="1U8ltCqRQHGSzhBnQ-hno4YLBDUwWJTlT", output=localizer_path, quiet=False)

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

        # 🔹 Classification Head
        self.localizer = VGG11Localizer(in_channels=in_channels, dropout_p=0.5)

        # 🔹 Segmentation branch 
        self.segmenter = VGG11UNet(num_classes=seg_classes, in_channels=in_channels, dropout_p=0.5)
        
        # 🔥 Load pretrained weights
        self._load_weights(classifier_path, localizer_path, unet_path)

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
            self.localizer.load_state_dict(torch.load(localizer_path, map_location=device))
            print("✅ Loaded localizer weights")
        except Exception as e:
            print(f"⚠️ Localizer load failed: {e}")


        try:
            self.segmenter.load_state_dict(torch.load(unet_path, map_location=device))
            print("✅ Loaded segmentation weights")
        except Exception as e:
            print(f"⚠️ Segmentation load failed: {e}")

    def forward(self, x: torch.Tensor):
        # 🔹 Encoder
        bottleneck = self.encoder(x)

        # 1️⃣ CLASSIFICATION
        cls_out = self.classifier_head(bottleneck)

        # 2️⃣ LOCALIZATION (Scaling Fix)
        loc_out = self.localizer(x) 

        # 3️⃣ SEGMENTATION (Skip-connection Decoder)
        seg_out = self.seg_head(d1)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }
