import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader."""

    def __init__(self, root_dir: str, split: str = "train", mask: bool = False):
        self.root_dir = root_dir
        self.split = split
        self.mask = mask

        self.image_dir = os.path.join(root_dir, "images")
        self.anno_dir = os.path.join(root_dir, "annotations", "xmls")
        self.trimap_dir = os.path.join(root_dir, "annotations", "trimaps")

        # Load image IDs and labels
        if split in ["train", "val"]:
            list_file = os.path.join(root_dir, "annotations", "trainval.txt")
        else:
            list_file = os.path.join(root_dir, "annotations", f"{split}.txt")

        with open(list_file, "r") as f:
            lines = f.readlines()
            self.image_ids = [line.strip().split(" ")[0] for line in lines]
            self.labels = [int(line.strip().split(" ")[1]) - 1 for line in lines]

        # Transformations
        # Note: interpolation=0 (Nearest Neighbor) is CRITICAL for masks to keep class indices 0,1,2
        transform_list = [
            A.Resize(224, 224, interpolation=0), 
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        if self.mask:
            self.transform = A.Compose(transform_list, additional_targets={'mask': 'mask'})
        else:
            self.transform = A.Compose(transform_list)

    def _load_bbox(self, xml_path):
        """Loads bbox and returns [cx, cy, w, h] in ORIGINAL pixel coordinates."""
        if not os.path.exists(xml_path):
            # Fallback for missing XMLs
            return torch.tensor([0.5, 0.5, 0.1, 0.1], dtype=torch.float32)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to (cx, cy, w, h)
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + ".jpg")

        # Load Image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        orig_h, orig_w = image_np.shape[:2]

        # Load Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Load Bounding Box
        xml_path = os.path.join(self.anno_dir, image_id + ".xml")
        bbox = self._load_bbox(xml_path)

        # ✅ CRITICAL FIX: Normalize bbox to [0, 1] based on original image dimensions
        # This makes the dataset compatible with the train script's scaling (bbox * 224)
        if orig_w > 0 and orig_h > 0:
            scale_vec = torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=torch.float32)
            bbox = bbox / scale_vec

        # Load segmentation mask if needed
        segmentation_mask = np.empty(0, dtype=np.uint8)
        if self.mask:
            mask_path = os.path.join(self.trimap_dir, image_id + ".png")
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask)
            # The trimap pixels are 1, 2, 3. Map them to 0, 1, 2 for CrossEntropy
            mask_np = np.clip(mask_np - 1, 0, 2)
            segmentation_mask = mask_np

        # Apply transformations
        if self.mask:
            transformed = self.transform(image=image_np, mask=segmentation_mask)
            image_tensor = transformed['image']
            mask_tensor = transformed['mask'].long()
        else:
            transformed = self.transform(image=image_np)
            image_tensor = transformed['image']
            mask_tensor = torch.empty(0, dtype=torch.long)

        return image_tensor, label, bbox, mask_tensor
