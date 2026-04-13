"""
Q2.5 helper script: log 10 localization predictions to W&B with GT box (green),
predicted box (red), IoU, and a confidence proxy.

Why a confidence proxy?
The provided VGG11Localizer regresses only 4 bbox values and does not output an
objectness/confidence score. For the report, this script uses the classifier's
maximum softmax probability on the predicted crop as a grounded proxy for
confidence of the localized object.
"""

import argparse
import os
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer
from models.classification import VGG11Classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized CHW tensor to HWC uint8-like float image."""
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def cxcywh_to_xyxy(box: torch.Tensor) -> Tuple[float, float, float, float]:
    cx, cy, w, h = [float(v) for v in box]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def clamp_xyxy(box_xyxy: Tuple[float, float, float, float], size: int = 224) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0.0, min(size - 1.0, x1))
    y1 = max(0.0, min(size - 1.0, y1))
    x2 = max(0.0, min(size - 1.0, x2))
    y2 = max(0.0, min(size - 1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def iou_xyxy(box1, box2, eps: float = 1e-6) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter
    return float(inter / (union + eps))


def draw_boxes(image_tensor: torch.Tensor, gt_box: torch.Tensor, pred_box: torch.Tensor, title: str = ""):
    img = denormalize(image_tensor)

    gt_xyxy = clamp_xyxy(cxcywh_to_xyxy(gt_box), 224)
    pr_xyxy = clamp_xyxy(cxcywh_to_xyxy(pred_box), 224)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)

    gx1, gy1, gx2, gy2 = gt_xyxy
    px1, py1, px2, py2 = pr_xyxy

    ax.add_patch(Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1,
                           linewidth=2, edgecolor='green', facecolor='none'))
    ax.add_patch(Rectangle((px1, py1), px2 - px1, py2 - py1,
                           linewidth=2, edgecolor='red', facecolor='none'))

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def crop_from_pred(image_tensor: torch.Tensor, pred_box: torch.Tensor) -> torch.Tensor:
    """
    Crop predicted bbox from normalized tensor image, resize back to 224x224
    for classifier confidence proxy.
    """
    x1, y1, x2, y2 = clamp_xyxy(cxcywh_to_xyxy(pred_box), 224)
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(max(x1 + 1, x2)), int(max(y1 + 1, y2))
    crop = image_tensor[:, y1i:y2i, x1i:x2i]
    if crop.numel() == 0:
        return image_tensor
    crop = F.interpolate(crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    return crop.squeeze(0)


def load_checkpoint_state(model: torch.nn.Module, ckpt_path: str, label: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"⚠️ {label} checkpoint not found: {ckpt_path}")
        return False

    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state, strict=True)
        print(f"✅ Loaded {label} checkpoint from {ckpt_path}")
        return True
    except Exception as e:
        print(f"⚠️ Could not strictly load {label}: {e}")
        try:
            model.load_state_dict(state, strict=False)
            print(f"✅ Loaded {label} checkpoint with strict=False from {ckpt_path}")
            return True
        except Exception as e2:
            print(f"❌ Failed to load {label}: {e2}")
            return False


def maybe_get_image_id(dataset, idx: int) -> str:
    if hasattr(dataset, "image_ids"):
        return str(dataset.image_ids[idx])
    return str(idx)


@torch.no_grad()
def run_q25(
    data_dir: str,
    localizer_ckpt: str,
    classifier_ckpt: str,
    split: str,
    num_samples: int,
    start_idx: int,
    project: str,
    run_name: str,
    seed: int,
):
    set_seed(seed)

    dataset = OxfordIIITPetDataset(root_dir=data_dir, split=split, mask=False)

    localizer = VGG11Localizer().to(DEVICE)
    classifier = VGG11Classifier(num_classes=37).to(DEVICE)

    load_checkpoint_state(localizer, localizer_ckpt, "localizer")
    load_checkpoint_state(classifier, classifier_ckpt, "classifier")

    localizer.eval()
    classifier.eval()

    indices = list(range(start_idx, min(start_idx + num_samples, len(dataset))))
    if len(indices) < num_samples:
        indices = list(range(min(num_samples, len(dataset))))

    wandb.init(project=project, name=run_name, config={
        "task": "Q2.5",
        "split": split,
        "num_samples": len(indices),
        "confidence_definition": "max softmax probability from classifier on predicted crop",
    })

    table = wandb.Table(columns=[
        "image_id",
        "label",
        "overlay",
        "confidence_score",
        "iou",
        "gt_box_cxcywh",
        "pred_box_cxcywh",
        "notes",
    ])

    ious: List[float] = []
    confs: List[float] = []

    for idx in indices:
        image, label, gt_box_norm, _ = dataset[idx]
        image_b = image.unsqueeze(0).to(DEVICE)

        pred_box = localizer(image_b).squeeze(0).cpu()  # already in pixel space [0,224]
        gt_box = (gt_box_norm.float() * 224.0).cpu()

        crop = crop_from_pred(image, pred_box).unsqueeze(0).to(DEVICE)
        logits = classifier(crop)
        probs = torch.softmax(logits, dim=1)
        confidence = float(probs.max(dim=1).values.item())

        gt_xyxy = clamp_xyxy(cxcywh_to_xyxy(gt_box), 224)
        pr_xyxy = clamp_xyxy(cxcywh_to_xyxy(pred_box), 224)
        iou = iou_xyxy(gt_xyxy, pr_xyxy)

        note = ""
        if confidence >= 0.80 and iou < 0.50:
            note = "High confidence, low IoU failure case"
        elif iou < 0.20:
            note = "Likely missed object / poor localization"
        elif iou >= 0.70:
            note = "Good localization"

        fig = draw_boxes(
            image, gt_box, pred_box,
            title=f"{maybe_get_image_id(dataset, idx)} | conf={confidence:.3f} | IoU={iou:.3f}"
        )
        overlay = wandb.Image(fig)
        plt.close(fig)

        table.add_data(
            maybe_get_image_id(dataset, idx),
            int(label),
            overlay,
            confidence,
            iou,
            [round(float(v), 2) for v in gt_box.tolist()],
            [round(float(v), 2) for v in pred_box.tolist()],
            note
        )

        ious.append(iou)
        confs.append(confidence)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.0

    # Pick a failure case automatically for easier report writing.
    failure_rank = None
    if ious:
        scored = sorted(
            [(i, confs[i], ious[i]) for i in range(len(ious))],
            key=lambda x: (-x[1], x[2])
        )
        failure_rank = scored[0]

    wandb.log({
        "q25_detection_table": table,
        "q25_mean_iou": mean_iou,
        "q25_mean_confidence": mean_conf,
    })

    if failure_rank is not None:
        idx_local, conf_bad, iou_bad = failure_rank
        wandb.summary["failure_case_table_row"] = int(idx_local)
        wandb.summary["failure_case_confidence"] = float(conf_bad)
        wandb.summary["failure_case_iou"] = float(iou_bad)

    wandb.finish()
    print("✅ Logged Q2.5 table to W&B.")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean confidence: {mean_conf:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Q2.5 localization W&B table logger")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--localizer-ckpt", type=str, default="checkpoints/localizer.pth")
    parser.add_argument("--classifier-ckpt", type=str, default="checkpoints/classifier.pth")
    parser.add_argument("--project", type=str, default="da6401_assignment2")
    parser.add_argument("--run-name", type=str, default="q25_detection_table")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_q25(
        data_dir=args.data_dir,
        localizer_ckpt=args.localizer_ckpt,
        classifier_ckpt=args.classifier_ckpt,
        split=args.split,
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        project=args.project,
        run_name=args.run_name,
        seed=args.seed,
    )
