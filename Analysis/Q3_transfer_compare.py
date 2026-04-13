import os
import time
import copy
import random
import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet
from models.classification import VGG11Classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CombinedSegLoss(nn.Module):
    def __init__(self, class_weights=(1.0, 5.0, 10.0), dice_weight: float = 2.0):
        super().__init__()
        self.register_buffer("weights", torch.tensor(class_weights, dtype=torch.float32))
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=self.weights)

    def dice_loss(self, logits: torch.Tensor, target: torch.Tensor, num_classes: int = 3):
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        inter = (probs * target_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2.0 * inter + 1e-6) / (union + 1e-6)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        ce = self.ce(logits, target)
        dice = self.dice_loss(logits, target, num_classes=logits.shape[1])
        total = ce + self.dice_weight * dice
        return total, ce.detach(), dice.detach()


def compute_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> Dict[str, float]:
    preds = logits.argmax(dim=1)
    pixel_acc = (preds == target).float().mean().item()

    pred_one_hot = F.one_hot(preds, num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    inter = (pred_one_hot * target_one_hot).sum(dim=(2, 3))
    union = pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = ((2.0 * inter + 1e-6) / (union + 1e-6)).mean().item()

    return {"pixel_acc": pixel_acc, "dice": dice}


def make_loaders(data_dir: str, batch_size: int, val_ratio: float = 0.15, seed: int = 42):
    dataset = OxfordIIITPetDataset(root_dir=data_dir, split="train", mask=True)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def maybe_load_pretrained_encoder(model: VGG11UNet, classifier_ckpt: str):
    if not classifier_ckpt or not os.path.exists(classifier_ckpt):
        print(f"⚠️ No classifier checkpoint found at {classifier_ckpt}. Running without pretrained encoder load.")
        return False
    try:
        clf = VGG11Classifier(num_classes=37).to(DEVICE)
        clf.load_state_dict(torch.load(classifier_ckpt, map_location=DEVICE))
        model.encoder.load_state_dict(clf.encoder.state_dict(), strict=False)
        print(f"✅ Loaded encoder weights from {classifier_ckpt}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to load encoder weights from {classifier_ckpt}: {e}")
        return False


def set_transfer_strategy(model: VGG11UNet, strategy: str):
    # First enable everything
    for p in model.parameters():
        p.requires_grad = True

    if strategy == "strict_feature_extractor":
        for p in model.encoder.parameters():
            p.requires_grad = False
        description = "Encoder fully frozen; only decoder + final head train."

    elif strategy == "partial_finetune":
        # Freeze early generic blocks, train later semantic blocks and decoder.
        for block in [model.encoder.block1, model.encoder.block2, model.encoder.block3]:
            for p in block.parameters():
                p.requires_grad = False
        description = "Blocks 1-3 frozen; blocks 4-5 + decoder train."

    elif strategy == "full_finetune":
        description = "Entire encoder + decoder train end-to-end."

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return description, trainable, total


def run_epoch(model, loader, criterion, optimizer=None) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = total_ce = total_dice_loss = 0.0
    total_pixel_acc = total_dice = 0.0
    steps = 0

    for images, _, _, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True).long()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss, ce, dloss = criterion(logits, masks)
            metrics = compute_metrics(logits, masks, num_classes=logits.shape[1])

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        total_loss += loss.item()
        total_ce += ce.item()
        total_dice_loss += dloss.item()
        total_pixel_acc += metrics["pixel_acc"]
        total_dice += metrics["dice"]
        steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "ce_loss": total_ce / max(steps, 1),
        "dice_loss": total_dice_loss / max(steps, 1),
        "pixel_acc": total_pixel_acc / max(steps, 1),
        "dice": total_dice / max(steps, 1),
    }


def train_one_strategy(args, strategy: str):
    run_name = f"{args.run_prefix}_{strategy}"
    wandb.init(project=args.project, name=run_name, config=vars(args) | {"strategy": strategy})

    train_loader, val_loader = make_loaders(args.data_dir, args.batch_size, args.val_ratio, args.seed)

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(DEVICE)
    pretrained_loaded = maybe_load_pretrained_encoder(model, args.classifier_ckpt)
    description, trainable, total = set_transfer_strategy(model, strategy)

    criterion = CombinedSegLoss(class_weights=(1.0, 5.0, 10.0), dice_weight=args.dice_weight).to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_dice = -1.0
    best_path = os.path.join(args.ckpt_dir, f"{run_name}.pth")

    wandb.summary["strategy_description"] = description
    wandb.summary["pretrained_encoder_loaded"] = pretrained_loaded
    wandb.summary["trainable_params"] = trainable
    wandb.summary["total_params"] = total

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_stats = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        epoch_time = time.perf_counter() - start
        val_stats = run_epoch(model, val_loader, criterion, optimizer=None)

        log_data = {
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            "train_loss": train_stats["loss"],
            "train_ce_loss": train_stats["ce_loss"],
            "train_dice_loss": train_stats["dice_loss"],
            "train_pixel_acc": train_stats["pixel_acc"],
            "train_dice": train_stats["dice"],
            "val_loss": val_stats["loss"],
            "val_ce_loss": val_stats["ce_loss"],
            "val_dice_loss": val_stats["dice_loss"],
            "val_pixel_acc": val_stats["pixel_acc"],
            "val_dice": val_stats["dice"],
            "generalization_gap_loss": val_stats["loss"] - train_stats["loss"],
            "generalization_gap_dice": train_stats["dice"] - val_stats["dice"],
        }
        wandb.log(log_data)
        print(
            f"[{strategy}] Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} | "
            f"train_dice={train_stats['dice']:.4f} val_dice={val_stats['dice']:.4f} | "
            f"time={epoch_time:.2f}s"
        )

        if val_stats["dice"] > best_dice:
            best_dice = val_stats["dice"]
            torch.save(model.state_dict(), best_path)
            wandb.summary["best_epoch"] = epoch
            wandb.summary["best_val_dice"] = best_dice
            wandb.summary["best_val_loss"] = val_stats["loss"]
            wandb.summary["best_val_pixel_acc"] = val_stats["pixel_acc"]
            wandb.summary["best_checkpoint"] = best_path

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Q2.3 Transfer learning comparison for segmentation")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--classifier-ckpt", type=str, default="checkpoints/classifier.pth")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--project", type=str, default="da6401_assignment2")
    parser.add_argument("--run-prefix", type=str, default="q23_transfer")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout-p", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=2.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    for strategy in ["strict_feature_extractor", "partial_finetune", "full_finetune"]:
        train_one_strategy(args, strategy)


if __name__ == "__main__":
    main()
