import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask=False):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.anno_dir = os.path.join(root_dir, "annotations", "xmls")
        self.trimap_dir = os.path.join(root_dir, "annotations", "trimaps")
        self.mask = mask

        # Default transform
        if transform is None:
            if mask:
                self.transform = A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
        else:
            self.transform = transform

        # Collect image ids from jpg files
        self.image_ids = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(self.image_dir)
            if f.endswith(".jpg")
        ])

        # Simple label extraction from filename prefix
        breeds = sorted(set("_".join(img_id.split("_")[:-1]) for img_id in self.image_ids))
        breed_to_idx = {breed: idx for idx, breed in enumerate(breeds)}
        self.labels = [breed_to_idx["_".join(img_id.split("_")[:-1])] for img_id in self.image_ids]

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

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        image_np = np.array(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        xml_path = os.path.join(self.anno_dir, image_id + ".xml")
        bbox = self._load_bbox(xml_path)

        # Scale bbox to resized 224x224 image space
        scale_x = 224.0 / orig_w
        scale_y = 224.0 / orig_h
        bbox = torch.tensor([
            bbox[0] * scale_x,
            bbox[1] * scale_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y
        ], dtype=torch.float32)

        segmentation_mask = np.empty((0,), dtype=np.uint8)
        if self.mask:
            mask_path = os.path.join(self.trimap_dir, image_id + ".png")
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask) - 1
            mask_np = np.clip(mask_np, 0, 2)
            segmentation_mask = mask_np

        if self.mask:
            transformed = self.transform(image=image_np, mask=segmentation_mask)
            image = transformed["image"]
            mask_tensor = transformed["mask"].long()
        else:
            transformed = self.transform(image=image_np)
            image = transformed["image"]
            mask_tensor = torch.empty(0, dtype=torch.long)

        return image, label, bbox, mask_tensor
