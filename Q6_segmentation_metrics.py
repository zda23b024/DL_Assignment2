import os
import time
import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class CrossEntropyLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets.long())


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Multi-class Dice for trimap classes.
    Ignores background class 0 and averages Dice over classes 1 and 2.
    logits: [B, C, H, W]
    targets: [B, H, W]
    """
    preds = torch.argmax(logits, dim=1)

    dice_scores = []
    for cls in [1, 2]:
        pred_c = (preds == cls).float()
        target_c = (targets == cls).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice)

    return torch.stack(dice_scores, dim=0).mean().item()


def pixel_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).float().mean()
    return correct.item()


def find_sample_images(dataset, num_samples: int = 5) -> List[int]:
    picks = []
    for idx in range(len(dataset)):
        _, _, _, mask = dataset[idx]
        if mask.sum().item() > 0:
            picks.append(idx)
        if len(picks) >= num_samples:
            break
    return picks


def denorm_image(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    return img


def prepare_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    return mask_tensor.detach().cpu().numpy().squeeze()


def visualize_triplet(image_t: torch.Tensor, gt_mask_t: torch.Tensor, pred_logits_t: torch.Tensor, title: str):
    pred_mask = torch.argmax(pred_logits_t, dim=0)

    image = denorm_image(image_t)
    gt_mask = prepare_mask(gt_mask_t)
    pred_mask_np = prepare_mask(pred_mask)

    fig = plt.figure(figsize=(10, 3.6))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(gt_mask, cmap="viridis")
    ax2.set_title("Ground Truth Trimap")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(pred_mask_np, cmap="viridis")
    ax3.set_title("Predicted Trimap Mask")
    ax3.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def extract_logits(outputs):
    if isinstance(outputs, dict):
        if "segmentation" in outputs:
            return outputs["segmentation"]
        return list(outputs.values())[-1]
    if isinstance(outputs, (tuple, list)):
        return outputs[-1]
    return outputs


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_acc = 0.0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            images, _, _, masks = batch
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).long()

            logits = extract_logits(model(images))
            loss = criterion(logits, masks)

            total_loss += loss.item()
            total_dice += dice_score_from_logits(logits, masks)
            total_acc += pixel_accuracy_from_logits(logits, masks)
            batches += 1

    return {
        "loss": total_loss / max(batches, 1),
        "dice": total_dice / max(batches, 1),
        "pixel_acc": total_acc / max(batches, 1),
    }


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_acc = 0.0
    batches = 0

    for batch in loader:
        images, _, _, masks = batch
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).long()

        optimizer.zero_grad()
        logits = extract_logits(model(images))
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score_from_logits(logits, masks)
        total_acc += pixel_accuracy_from_logits(logits, masks)
        batches += 1

    return {
        "loss": total_loss / max(batches, 1),
        "dice": total_dice / max(batches, 1),
        "pixel_acc": total_acc / max(batches, 1),
    }


def log_sample_predictions(model, dataset, sample_indices, prefix="q26"):
    model.eval()
    logged = 0

    with torch.no_grad():
        for idx in sample_indices:
            image_t, _, _, mask = dataset[idx]
            image = image_t.unsqueeze(0).to(DEVICE)

            logits = extract_logits(model(image))
            fig = visualize_triplet(image_t, mask, logits[0].cpu(), title=f"Sample {idx}")

            wandb.log({f"{prefix}_sample_{logged + 1}": wandb.Image(fig)})
            plt.close(fig)
            logged += 1


def main():
    parser = argparse.ArgumentParser(description="Q2.6 Segmentation metrics and qualitative logging")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--encoder-ckpt", type=str, default="checkpoints/classifier.pth")
    parser.add_argument("--run-name", type=str, default="q26_segmentation_metrics")
    parser.add_argument("--project", type=str, default="da6401_assignment2")
    args = parser.parse_args()

    set_seed(args.seed)
    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    full_dataset = OxfordIIITPetDataset(root_dir=args.data_dir, split="train", mask=True)

    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    try:
        model = VGG11UNet(pretrained=False).to(DEVICE)
    except TypeError:
        model = VGG11UNet().to(DEVICE)

    if os.path.exists(args.encoder_ckpt):
        try:
            ckpt = torch.load(args.encoder_ckpt, map_location=DEVICE)
            state_dict = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint with missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"Warning: Could not load encoder checkpoint: {e}")

    criterion = CrossEntropyLossWrapper()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_dice = -1.0
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"{args.run_name}.pth")

    for epoch in range(args.epochs):
        start = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion)

        epoch_time = time.time() - start

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_dice": val_metrics["dice"],
            "train_pixel_acc": train_metrics["pixel_acc"],
            "val_pixel_acc": val_metrics["pixel_acc"],
            "loss_gap": val_metrics["loss"] - train_metrics["loss"],
            "dice_gap": train_metrics["dice"] - val_metrics["dice"],
            "pixel_acc_gap": train_metrics["pixel_acc"] - val_metrics["pixel_acc"],
            "epoch_time_sec": epoch_time,
        })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"train_dice={train_metrics['dice']:.4f} val_dice={val_metrics['dice']:.4f} | "
            f"train_acc={train_metrics['pixel_acc']:.4f} val_acc={val_metrics['pixel_acc']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1}, ckpt_path)

    val_base_dataset = val_dataset.dataset if hasattr(val_dataset, "dataset") else full_dataset
    if hasattr(val_dataset, "indices"):
        candidate_indices = val_dataset.indices
        subset_for_picks = Subset(val_base_dataset, candidate_indices)
        local_picks = find_sample_images(subset_for_picks, num_samples=args.sample_count)
        sample_indices = [candidate_indices[i] for i in local_picks]
    else:
        sample_indices = find_sample_images(val_base_dataset, num_samples=args.sample_count)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    log_sample_predictions(model, full_dataset, sample_indices, prefix="q26")

    wandb.summary["best_val_dice"] = best_val_dice
    wandb.summary["q26_note"] = (
        "Compare val_pixel_acc vs val_dice over early epochs; pixel accuracy often appears higher due to background dominance."
    )

    wandb.finish()


if __name__ == "__main__":
    main()
