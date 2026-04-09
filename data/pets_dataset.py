import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Returns:
        image: Tensor [3,H,W] normalized to [0,1]
        label: Tensor scalar in [0,36]
        bbox: Tensor [4] as [x_center, y_center, width, height] normalized [0,1]
        mask: Tensor [H,W] with classes 0=pet,1=border,2=background
    """

    def __init__(self, root="data/oxford_pet", split="trainval", image_size=224, mask=False):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.mask = mask

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

        # Oxford-IIIT Pet split files may have header lines; skip lines with fewer than 2 columns
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_name = parts[0]
            class_id = int(parts[1]) - 1  # convert 1..37 -> 0..36
            image_path = os.path.join(self.images_dir, image_name + ".jpg")
            xml_path = os.path.join(self.xml_dir, image_name + ".xml")
            mask_path = os.path.join(self.trimap_dir, image_name + ".png")
            if os.path.exists(image_path) and os.path.exists(xml_path):
                self.samples.append((image_name, class_id))

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found. Check dataset paths.")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_path):
        """Load image, resize to image_size, convert to tensor [C,H,W] normalized to [0,1]."""
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
        return image, orig_w, orig_h

    def _load_bbox(self, xml_path, orig_w, orig_h):
        """Load bbox from XML and return normalized [cx,cy,w,h] in [0,1]"""
        if not os.path.exists(xml_path):
            return torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        bbox = root.find("object").find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # convert to center x,y and width,height
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin

        # scale to resized image
        cx *= self.image_size / orig_w
        cy *= self.image_size / orig_h
        w *= self.image_size / orig_w
        h *= self.image_size / orig_h

        # normalize to [0,1]
        cx /= self.image_size
        cy /= self.image_size
        w /= self.image_size
        h /= self.image_size

        return torch.tensor([cx, cy, w, h], dtype=torch.float32)

    def _load_mask(self, mask_path):
        """Load trimap mask, resize, remap classes [1,2,3] -> [0,1,2]."""
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)
        remapped = np.zeros_like(mask_np, dtype=np.int64)
        remapped[mask_np == 1] = 0  # pet
        remapped[mask_np == 2] = 1  # border
        remapped[mask_np == 3] = 2  # background
        return torch.tensor(remapped, dtype=torch.long)

    def __getitem__(self, idx):
        image_name, class_id = self.samples[idx]
        image_path = os.path.join(self.images_dir, image_name + ".jpg")
        xml_path = os.path.join(self.xml_dir, image_name + ".xml")
        mask_path = os.path.join(self.trimap_dir, image_name + ".png")

        # Load image and original size
        image, orig_w, orig_h = self._load_image(image_path)

        # Load bbox normalized
        bbox = self._load_bbox(xml_path, orig_w, orig_h)

        # Load segmentation mask if required
        if self.mask:
            mask = self._load_mask(mask_path)
        else:
            mask = torch.empty(0, dtype=torch.long)

        label = torch.tensor(class_id, dtype=torch.long)

        return image, label, bbox, mask
