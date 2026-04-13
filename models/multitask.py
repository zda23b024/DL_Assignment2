"""
Unified multi-task model that loads pretrained single-task checkpoints.
"""

import os
import gdown
import torch
import torch.nn as nn

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

        if not os.path.exists(classifier_path):
            gdown.download(id="1qavuPzFvrWYyLsk6SnNS843S9RWYgje7", output=classifier_path, quiet=False)

        if not os.path.exists(localizer_path):
            gdown.download(id="1U8ltCqRQHGSzhBnQ-hno4YLBDUwWJTlT", output=localizer_path, quiet=False)

        if not os.path.exists(unet_path):
            gdown.download(id="1uOfQ1X5al6Kwjp9r6H1z6aENeU9oa7h9", output=unet_path, quiet=False)

        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),
            nn.Linear(4096, num_breeds),
        )

        # Keep single-task submodules and reuse their forward exactly.
        self.localizer = VGG11Localizer(in_channels=in_channels, dropout_p=0.5)
        self.segmenter = VGG11UNet(num_classes=seg_classes, in_channels=in_channels, dropout_p=0.5)

        self._load_weights(classifier_path, localizer_path, unet_path)

    @staticmethod
    def _extract_state_dict(checkpoint_obj):
        """Support both raw state_dict and wrapped checkpoints."""
        if isinstance(checkpoint_obj, dict):
            if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
                return checkpoint_obj["state_dict"]
            if "model_state_dict" in checkpoint_obj and isinstance(checkpoint_obj["model_state_dict"], dict):
                return checkpoint_obj["model_state_dict"]
        return checkpoint_obj

    def _load_weights(self, classifier_path, localizer_path, unet_path):
        device = next(self.parameters()).device

        try:
            from models.classification import VGG11Classifier

            classifier = VGG11Classifier(num_classes=37).to(device)
            ckpt = torch.load(classifier_path, map_location=device)
            classifier.load_state_dict(self._extract_state_dict(ckpt), strict=False)
            self.encoder.load_state_dict(classifier.encoder.state_dict(), strict=False)
            self.classifier_head.load_state_dict(classifier.classifier.state_dict(), strict=False)
            print("✅ Loaded classifier weights")
        except Exception as e:
            print(f"⚠️ Classifier load failed: {e}")

        try:
            ckpt = torch.load(localizer_path, map_location=device)
            self.localizer.load_state_dict(self._extract_state_dict(ckpt), strict=False)
            print("✅ Loaded localizer weights")
        except Exception as e:
            print(f"⚠️ Localizer load failed: {e}")

        try:
            ckpt = torch.load(unet_path, map_location=device)
            self.segmenter.load_state_dict(self._extract_state_dict(ckpt), strict=False)
            print("✅ Loaded segmentation weights")
        except Exception as e:
            print(f"⚠️ Segmentation load failed: {e}")

    def forward(self, x: torch.Tensor):
        bottleneck = self.encoder(x)

        cls_out = self.classifier_head(bottleneck)
        loc_out = self.localizer(x)
        seg_out = self.segmenter(x)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }
