import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


class PetsDataset(Dataset):
    def __init__(self, image_dir, anno_dir, trimap_dir, image_ids, labels, transform=None, mask=False):
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.trimap_dir = trimap_dir
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform
        self.mask = mask

    def __len__(self):
        return len(self.image_ids)

    def _load_bbox(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        obj = root.find("object")
        bndbox = obj.find("bndbox")

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Convert to [cx, cy, w, h]
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin

        return [cx, cy, w, h]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + ".jpg")

        # Load image
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        image_np = np.array(image)

        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Load bbox in original image space
        xml_path = os.path.join(self.anno_dir, image_id + ".xml")
        bbox = self._load_bbox(xml_path)

        # Scale bbox to resized 224x224 image space
        scale_x = 224.0 / orig_w
        scale_y = 224.0 / orig_h

        bbox = torch.tensor([
            bbox[0] * scale_x,  # cx
            bbox[1] * scale_y,  # cy
            bbox[2] * scale_x,  # w
            bbox[3] * scale_y   # h
        ], dtype=torch.float32)

        # Load segmentation mask if needed
        segmentation_mask = np.empty(0, dtype=np.uint8)
        if self.mask:
            mask_path = os.path.join(self.trimap_dir, image_id + ".png")
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask) - 1
            mask_np = np.clip(mask_np, 0, 2)
            segmentation_mask = mask_np

        # Apply transforms
        if self.mask:
            transformed = self.transform(image=image_np, mask=segmentation_mask)
            image = transformed["image"]
            mask_tensor = transformed["mask"].long()
        else:
            transformed = self.transform(image=image_np)
            image = transformed["image"]
            mask_tensor = torch.empty(0, dtype=torch.long)

        return image, label, bbox, mask_tensor
