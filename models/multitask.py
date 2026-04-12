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

        # Shared encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Classification head
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

        # Localization head
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
        self.loc_activation = nn.Sigmoid()

        # Segmentation decoder
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
        # Load classifier weights
        try:
            classifier_ckpt = torch.load(classifier_path, map_location="cpu")
            self.encoder.load_state_dict(
                {k.replace("encoder.", ""): v for k, v in classifier_ckpt.items() if k.startswith("encoder.")},
                strict=False
            )
            self.classifier_head.load_state_dict(
                {k.replace("classifier.", ""): v for k, v in classifier_ckpt.items() if k.startswith("classifier.")},
                strict=False
            )
            print("✅ Loaded classifier weights")
        except Exception as e:
            print(f"⚠️ Could not load classifier weights: {e}")

        # Load localizer weights
        try:
            localizer_ckpt = torch.load(localizer_path, map_location="cpu")
            self.encoder.load_state_dict(
                {k.replace("encoder.", ""): v for k, v in localizer_ckpt.items() if k.startswith("encoder.")},
                strict=False
            )
            self.localization_head.load_state_dict(
                {k.replace("regressor.", ""): v for k, v in localizer_ckpt.items() if k.startswith("regressor.")},
                strict=False
            )
            print("✅ Loaded localizer weights")
        except Exception as e:
            print(f"⚠️ Could not load localizer weights: {e}")

        # Load segmentation weights
        try:
            unet_ckpt = torch.load(unet_path, map_location="cpu")
            self.encoder.load_state_dict(
                {k.replace("encoder.", ""): v for k, v in unet_ckpt.items() if k.startswith("encoder.")},
                strict=False
            )
            self.up5.load_state_dict(
                {k.replace("up5.", ""): v for k, v in unet_ckpt.items() if k.startswith("up5.")},
                strict=False
            )
            self.dec5.load_state_dict(
                {k.replace("dec5.", ""): v for k, v in unet_ckpt.items() if k.startswith("dec5.")},
                strict=False
            )
            self.up4.load_state_dict(
                {k.replace("up4.", ""): v for k, v in unet_ckpt.items() if k.startswith("up4.")},
                strict=False
            )
            self.dec4.load_state_dict(
                {k.replace("dec4.", ""): v for k, v in unet_ckpt.items() if k.startswith("dec4.")},
                strict=False
            )
            self.up3.load_state_dict(
                {k.replace("up3.", ""): v for k, v in unet_ckpt.items() if k.startswith("up3.")},
                strict=False
            )
            self.dec3.load_state_dict(
                {k.replace("dec3.", ""): v for k, v in unet_ckpt.items() if k.startswith("dec3.")},
                strict=False
            )
            self.up2.load_state_dict(
                {k.replace("up2.", ""): v for k, v in unet_ckpt.items() if k.startswith("up2.")},
                strict=False
            )
            self.dec2.load_state_dict(
                {k.replace("dec2.", ""): v for k, v in unet_ckpt.items() if k.startswith("dec2.")},
                strict=False
            )
            self.up1.load_state_dict(
                {k.replace("up1.", ""): v for k, v in unet_ckpt.items() if k.startswith("up1.")},
                strict=False
            )
            self.dec1.load_state_dict(
                {k.replace("dec1.", ""): v for k, v in unet_ckpt.items() if k.startswith("dec1.")},
                strict=False
            )
            self.seg_head.load_state_dict(
                {k.replace("final.", ""): v for k, v in unet_ckpt.items() if k.startswith("final.")},
                strict=False
            )
            print("✅ Loaded segmentation weights")
        except Exception as e:
            print(f"⚠️ Could not load segmentation weights: {e}")

    def forward(self, x: torch.Tensor):
        bottleneck, feats = self.encoder(x, return_features=True)

        # Classification
        cls_out = self.classifier_head(bottleneck)

        # Localization
        loc_out = self.localization_head(bottleneck)
        loc_out = self.loc_activation(loc_out) * 224.0

        # Segmentation
        f1 = feats["block1"]
        f2 = feats["block2"]
        f3 = feats["block3"]
        f4 = feats["block4"]
        f5 = feats["block5"]

        s = self.up5(bottleneck)
        s = torch.cat([s, f5], dim=1)
        s = self.dec5(s)

        s = self.up4(s)
        s = torch.cat([s, f4], dim=1)
        s = self.dec4(s)

        s = self.up3(s)
        s = torch.cat([s, f3], dim=1)
        s = self.dec3(s)

        s = self.up2(s)
        s = torch.cat([s, f2], dim=1)
        s = self.dec2(s)

        s = self.up1(s)
        s = torch.cat([s, f1], dim=1)
        s = self.dec1(s)

        seg_out = self.seg_head(s)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "unet": seg_out,
        }
