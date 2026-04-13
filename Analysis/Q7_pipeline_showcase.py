"""
Q2.7 helper: Run the full pipeline on 3 novel pet images from the internet (or local files)
and log a W&B showcase.

This script does NOT modify the original project code.
It reuses the user's existing models and checkpoints:
- models/classification.py  -> VGG11Classifier
- models/localization.py    -> VGG11Localizer
- models/segmentation.py    -> VGG11UNet

What it logs to W&B:
- Original image
- Bounding-box overlay
- Predicted segmentation mask
- Segmentation overlay
- Cropped subject from predicted box
- Predicted breed index and confidence
- A small W&B table for all 3 images

Usage examples:
    python train_q27_pipeline_showcase.py \
        --image-urls "https://..." "https://..." "https://..." \
        --classifier-ckpt checkpoints/classifier.pth \
        --localizer-ckpt checkpoints/localizer.pth \
        --segmenter-ckpt checkpoints/unet.pth

    python train_q27_pipeline_showcase.py \
        --image-paths sample1.jpg sample2.jpg sample3.jpg
"""

import argparse
import io
import os
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torchvision import transforms

import wandb

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Q2.7 Pipeline Showcase on 3 novel pet images")
    parser.add_argument("--image-urls", nargs="*", default=[],
                        help="Three image URLs from the internet")
    parser.add_argument("--image-paths", nargs="*", default=[],
                        help="Three local image paths")
    parser.add_argument("--classifier-ckpt", type=str, default="checkpoints/classifier.pth")
    parser.add_argument("--localizer-ckpt", type=str, default="checkpoints/localizer.pth")
    parser.add_argument("--segmenter-ckpt", type=str, default="checkpoints/unet.pth")
    parser.add_argument("--run-name", type=str, default="q27_final_pipeline_showcase")
    parser.add_argument("--project", type=str, default="da6401_assignment2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def load_image_from_url(url: str, timeout: int = 20) -> Image.Image:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return img


def load_image_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    x = tensor.detach().cpu().permute(1, 2, 0).numpy()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = np.clip(x, 0.0, 1.0)
    return x


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (denormalize(tensor) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    3-class trimap coloring:
      0 -> black (background)
      1 -> green (pet)
      2 -> red (border)
    """
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask == 0] = [0, 0, 0]
    out[mask == 1] = [0, 255, 0]
    out[mask == 2] = [255, 0, 0]
    return out


def overlay_mask_on_image(image_pil: Image.Image, mask: np.ndarray, alpha: float = 0.35) -> Image.Image:
    base = np.array(image_pil).astype(np.float32)
    color_mask = mask_to_color(mask).astype(np.float32)
    out = ((1.0 - alpha) * base + alpha * color_mask).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def clamp_bbox_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def bbox_cxcywh_to_xyxy(bbox):
    cx, cy, bw, bh = bbox
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return x1, y1, x2, y2


def draw_bbox(image_pil: Image.Image, bbox_xyxy, color=(255, 0, 0), width=3, text=None) -> Image.Image:
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox_xyxy
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if text:
        draw.text((x1 + 3, max(0, y1 - 12)), text, fill=color)
    return img


def crop_with_bbox(image_pil: Image.Image, bbox_xyxy) -> Image.Image:
    x1, y1, x2, y2 = bbox_xyxy
    return image_pil.crop((x1, y1, x2, y2))


def load_models(args):
    device = args.device

    classifier = VGG11Classifier(num_classes=37).to(device)
    if os.path.exists(args.classifier_ckpt):
        classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
    else:
        raise FileNotFoundError(f"Classifier checkpoint not found: {args.classifier_ckpt}")
    classifier.eval()

    localizer = VGG11Localizer().to(device)
    if os.path.exists(args.localizer_ckpt):
        localizer.load_state_dict(torch.load(args.localizer_ckpt, map_location=device))
    else:
        raise FileNotFoundError(f"Localizer checkpoint not found: {args.localizer_ckpt}")
    localizer.eval()

    segmenter = VGG11UNet(num_classes=3).to(device)
    if os.path.exists(args.segmenter_ckpt):
        segmenter.load_state_dict(torch.load(args.segmenter_ckpt, map_location=device))
    else:
        raise FileNotFoundError(f"Segmenter checkpoint not found: {args.segmenter_ckpt}")
    segmenter.eval()

    return classifier, localizer, segmenter


@torch.no_grad()
def predict_one(classifier, localizer, segmenter, pil_img, transform, device):
    x = transform(pil_img).unsqueeze(0).to(device)

    cls_logits = classifier(x)
    loc_out = localizer(x)
    seg_logits = segmenter(x)

    probs = F.softmax(cls_logits, dim=1)
    conf, pred_breed = probs.max(dim=1)

    bbox = loc_out.squeeze(0).detach().cpu().numpy()
    mask = torch.argmax(seg_logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

    return {
        "input_tensor": x.squeeze(0).cpu(),
        "pred_breed": int(pred_breed.item()),
        "confidence": float(conf.item()),
        "bbox_cxcywh": bbox,
        "pred_mask": mask,
    }


def collect_sources(args):
    sources = []
    for url in args.image_urls:
        sources.append(("url", url))
    for path in args.image_paths:
        sources.append(("path", path))
    if len(sources) < 3:
        raise ValueError("Provide at least 3 images using --image-urls and/or --image-paths.")
    return sources[:3]


def load_source(source_kind, value, timeout=20):
    if source_kind == "url":
        return load_image_from_url(value, timeout=timeout), value
    return load_image_from_path(value), value


def main():
    args = parse_args()
    device = args.device
    transform = build_transform()

    classifier, localizer, segmenter = load_models(args)

    wandb.init(
        project=args.project,
        name=args.run_name,
        config={
            "question": "2.7",
            "classifier_ckpt": args.classifier_ckpt,
            "localizer_ckpt": args.localizer_ckpt,
            "segmenter_ckpt": args.segmenter_ckpt,
        },
    )

    table = wandb.Table(columns=[
        "source",
        "original",
        "bbox_overlay",
        "mask",
        "mask_overlay",
        "predicted_crop",
        "predicted_breed_idx",
        "confidence",
        "bbox_xyxy",
    ])

    sources = collect_sources(args)

    for idx, (kind, value) in enumerate(sources, start=1):
        pil_img, source_name = load_source(kind, value, timeout=args.timeout)
        pil_resized = pil_img.resize((224, 224))

        out = predict_one(classifier, localizer, segmenter, pil_resized, transform, device)

        bbox_xyxy = bbox_cxcywh_to_xyxy(out["bbox_cxcywh"])
        x1, y1, x2, y2 = clamp_bbox_xyxy(*bbox_xyxy, 224, 224)
        bbox_xyxy = (x1, y1, x2, y2)

        original = pil_resized
        bbox_overlay = draw_bbox(
            original,
            bbox_xyxy,
            color=(255, 0, 0),
            width=3,
            text=f"breed={out['pred_breed']} conf={out['confidence']:.3f}"
        )
        mask_pil = Image.fromarray(mask_to_color(out["pred_mask"]))
        mask_overlay = overlay_mask_on_image(original, out["pred_mask"], alpha=0.35)
        predicted_crop = crop_with_bbox(original, bbox_xyxy)

        wandb.log({
            f"q27_original_{idx}": wandb.Image(original, caption=f"Source: {source_name}"),
            f"q27_bbox_overlay_{idx}": wandb.Image(bbox_overlay, caption="Predicted bounding box"),
            f"q27_mask_{idx}": wandb.Image(mask_pil, caption="Predicted trimap"),
            f"q27_mask_overlay_{idx}": wandb.Image(mask_overlay, caption="Predicted segmentation overlay"),
            f"q27_crop_{idx}": wandb.Image(predicted_crop, caption="Crop from predicted box"),
            f"q27_pred_breed_idx_{idx}": out["pred_breed"],
            f"q27_confidence_{idx}": out["confidence"],
        })

        table.add_data(
            source_name,
            wandb.Image(original),
            wandb.Image(bbox_overlay),
            wandb.Image(mask_pil),
            wandb.Image(mask_overlay),
            wandb.Image(predicted_crop),
            out["pred_breed"],
            out["confidence"],
            str(bbox_xyxy),
        )

    wandb.log({"q27_pipeline_showcase_table": table})
    wandb.finish()
    print("Done. Check W&B for the Q2.7 showcase.")


if __name__ == "__main__":
    main()
