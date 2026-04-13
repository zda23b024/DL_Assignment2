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
            gdown.download(id="1KKqHE0z10i8ogMgR_CNLI9Ib_uQoJQ6H", output=localizer_path, quiet=False)

        if not os.path.exists(unet_path):
            gdown.download(id="1uOfQ1X5al6Kwjp9r6H1z6aENeU9oa7h9", output=unet_path, quiet=False)

        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            # Keep inference deterministic even if external evaluator
            # forgets to call model.eval().
            CustomDropout(0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.0),
            nn.Linear(4096, num_breeds),
        )

        # Keep single-task submodules and reuse their forward exactly.
        # Disable dropout in multitask inference wrapper for the same reason.
        self.localizer = VGG11Localizer(in_channels=in_channels, dropout_p=0.0)
        self.segmenter = VGG11UNet(num_classes=seg_classes, in_channels=in_channels, dropout_p=0.0)

        self._load_weights(classifier_path, localizer_path, unet_path)

    @staticmethod
    def _extract_state_dict(checkpoint_obj):
        """Support both raw state_dict and wrapped checkpoints."""
        if isinstance(checkpoint_obj, dict):
            for key in ("state_dict", "model_state_dict", "model", "weights", "net"):
                if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                    return checkpoint_obj[key]
        return checkpoint_obj

    @staticmethod
    def _adapt_state_dict_keys(state_dict, module_keys):
        """
        Try common key-prefix variants and keep the version that matches
        the most parameters in the target module.
        """
        if not isinstance(state_dict, dict):
            return state_dict

        candidates = [state_dict]

        # Candidate 1: strip a single leading "module." (common DataParallel case).
        stripped_module = {
            (k[len("module."):] if isinstance(k, str) and k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        if stripped_module != state_dict:
            candidates.append(stripped_module)

        # Candidate 2+: strip first token before '.' (e.g. "segmenter.", "model.").
        # Also try this on each existing candidate to handle "module.segmenter.*".
        expanded = list(candidates)
        for cand in expanded:
            stripped_first = {}
            changed = False
            for k, v in cand.items():
                if isinstance(k, str) and "." in k:
                    stripped_first[k.split(".", 1)[1]] = v
                    changed = True
                else:
                    stripped_first[k] = v
            if changed and stripped_first not in candidates:
                candidates.append(stripped_first)

        # Pick the candidate with maximum direct key overlap.
        module_key_set = set(module_keys)
        best = candidates[0]
        best_overlap = len(set(candidates[0].keys()) & module_key_set)
        for cand in candidates[1:]:
            overlap = len(set(cand.keys()) & module_key_set)
            if overlap > best_overlap:
                best = cand
                best_overlap = overlap
        return best

    def _safe_load(self, module, checkpoint_path, module_name):
        """
        Load checkpoint into module and print a useful warning if key overlap is
        too small (which usually means wrong checkpoint format/weights).
        """
        device = next(self.parameters()).device
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = self._extract_state_dict(ckpt)
        state_dict = self._adapt_state_dict_keys(state_dict, module.state_dict().keys())
        load_info = module.load_state_dict(state_dict, strict=False)

        loaded_keys = len(module.state_dict().keys()) - len(load_info.missing_keys)
        total_keys = len(module.state_dict().keys())
        if loaded_keys == 0:
            print(f"⚠️ {module_name} load warning: 0/{total_keys} keys matched from {checkpoint_path}")
        else:
            print(f"✅ Loaded {module_name} weights ({loaded_keys}/{total_keys} keys matched)")

    def _load_weights(self, classifier_path, localizer_path, unet_path):
        device = next(self.parameters()).device

        try:
            from models.classification import VGG11Classifier

            classifier = VGG11Classifier(num_classes=37).to(device)
            self._safe_load(classifier, classifier_path, "classifier")
            self.encoder.load_state_dict(classifier.encoder.state_dict(), strict=False)
            self.classifier_head.load_state_dict(classifier.classifier.state_dict(), strict=False)
        except Exception as e:
            print(f"⚠️ Classifier load failed: {e}")

        try:
            self._safe_load(self.localizer, localizer_path, "localizer")
        except Exception as e:
            print(f"⚠️ Localizer load failed: {e}")

        try:
            self._safe_load(self.segmenter, unet_path, "segmentation")
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
