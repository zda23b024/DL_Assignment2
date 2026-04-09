"""Dataset for Oxford-IIIT Pet."""

import os
import xml.etree.ElementTree as ET

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Returns:
        image: Tensor of shape [3, H, W]
        label: Tensor scalar in [0, 36]
        bbox: Tensor of shape [4] as [x_center, y_center, width, height] in pixel space
        mask: Tensor of shape [H, W] with classes:
              0 = pet, 1 = border, 2 = background
    """

    def __init__(self, root="data/oxford_pet", split="trainval", image_size=224):
        self.root = root
        self.split = split
        self.image_size = image_size

        self.images_dir = os.path.join(root, "images")
        self.annotations_dir = os.path.join(root, "annotations")
        self.xml_dir = os.path.join(self.annotations_dir, "xmls")
        self.trimap_dir = os.path.join(self.annotations_dir, "trimaps")

        split_file = os.path.join(self.annotations_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.samples = []

        with open(split_file, "r") as f:
            lines = f.readlines()

        # Oxford-IIIT Pet split files have 6 header lines
        for line in lines[6:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            image_name = parts[0]
            class_id = int(parts[1]) - 1  # convert 1..37 to 0..36

            image_path = os.path.join(self.images_dir, image_name + ".jpg")
            xml_path = os.path.join(self.xml_dir, image_name + ".xml")
            mask_path = os.path.join(self.trimap_dir, image_name + ".png")

            if os.path.exists(image_path) and os.path.exists(xml_path) and os.path.exists(mask_path):
                self.samples.append((image_name, class_id))

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found. Check dataset paths.")

    def __len__(self):
        return len(self.samples)

    def _load_bbox(self, xml_path, orig_w, orig_h):
        """Load bbox from XML and convert to pixel-space [xc, yc, w, h]."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        # scale bbox to resized image size
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        x_center *= scale_x
        y_center *= scale_y
        width *= scale_x
        height *= scale_y

        return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

    def _load_mask(self, mask_path):
        """Load trimap mask and remap classes from [1,2,3] -> [0,1,2]."""
        mask = Image.open(mask_path)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)

        # Oxford trimap values: 1=pet, 2=border, 3=background
        remapped = np.zeros_like(mask, dtype=np.int64)
        remapped[mask == 1] = 0
        remapped[mask == 2] = 1
        remapped[mask == 3] = 2

        return torch.tensor(remapped, dtype=torch.long)

    def _load_image(self, image_path):
        """Load image, resize, convert to tensor, normalize to [0,1]."""
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [H,W,C] -> [C,H,W]

        return image, orig_w, orig_h

    def __getitem__(self, idx):
        image_name, label = self.samples[idx]

        image_path = os.path.join(self.images_dir, image_name + ".jpg")
        xml_path = os.path.join(self.xml_dir, image_name + ".xml")
        mask_path = os.path.join(self.trimap_dir, image_name + ".png")

        image, orig_w, orig_h = self._load_image(image_path)
        bbox = self._load_bbox(xml_path, orig_w, orig_h)
        mask = self._load_mask(mask_path)
        label = torch.tensor(label, dtype=torch.long)

        return image, label, bbox, mask
