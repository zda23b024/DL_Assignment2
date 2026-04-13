"""
Q2.2 helper script for DA6401 Assignment 2.

Purpose:
- Keep the user's existing project files untouched.
- Reuse the same Oxford-IIIT Pet dataset loader and CustomDropout layer.
- Train 3 classification runs for dropout comparison:
    1) p = 0.0
    2) p = 0.2
    3) p = 0.5
- Log train/val loss and accuracy to Weights & Biases so the user can overlay curves.
- Log the generalization gap explicitly.

Expected project structure (run this from the project root):
    data/pets_dataset.py
    models/layers.py
    train_q22_dropout_compare.py

Example:
    python train_q22_dropout_compare.py --data-dir data --epochs 15 --batch-size 32 --lr 1e-4
"""

import argparse
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.layers import CustomDropout


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VGG11EncoderQ22(nn.Module):
    """Standalone VGG11 encoder with BatchNorm kept ON, matching the user's main flow."""

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)
        x = self.block5(x)
        x = self.pool5(x)
        return x


class VGG11ClassifierQ22(nn.Module):
    """Standalone classifier so the original files remain unchanged."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11EncoderQ22(in_channels=in_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)
        return x


def build_loaders(data_dir: str, batch_size: int, val_ratio: float, seed: int, num_workers: int):
    dataset = OxfordIIITPetDataset(root_dir=data_dir, split="train", mask=False)

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels, _, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return {"loss": avg_loss, "acc": avg_acc}


def train_one_setting(
    data_dir: str,
    dropout_p: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_ratio: float,
    seed: int,
    num_workers: int,
    project: str,
    run_prefix: str,
    save_dir: str,
) -> Dict[str, float]:
    set_seed(seed)
    train_loader, val_loader = build_loaders(data_dir, batch_size, val_ratio, seed, num_workers)

    model = VGG11ClassifierQ22(dropout_p=dropout_p).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    run_name = f"{run_prefix}_dropout_{dropout_p:.1f}"
    wandb.init(
        project=project,
        name=run_name,
        config={
            "task": "q2.2_dropout_internal_dynamics",
            "dropout_p": dropout_p,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "val_ratio": val_ratio,
            "seed": seed,
            "device": DEVICE,
        },
        reinit=True,
    )

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        for images, labels, _, _ in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_train_correct += (preds == labels).sum().item()
            total_train_samples += images.size(0)

        train_loss = total_train_loss / max(1, total_train_samples)
        train_acc = total_train_correct / max(1, total_train_samples)

        val_metrics = evaluate(model, val_loader, criterion)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]

        # Positive value means train loss < val loss, a simple overfitting proxy.
        generalization_gap = val_loss - train_loss

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "generalization_gap": generalization_gap,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"[{run_name}] Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} gap={generalization_gap:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "best_val_loss": val_loss,
                "best_val_acc": val_acc,
                "best_epoch": epoch,
                "dropout_p": dropout_p,
            }
            ckpt_path = os.path.join(save_dir, f"{run_name}.pth")
            torch.save(model.state_dict(), ckpt_path)

    summary_table = wandb.Table(
        columns=["dropout_p", "best_epoch", "best_val_loss", "best_val_acc"],
        data=[[dropout_p, best_metrics.get("best_epoch", -1), best_metrics.get("best_val_loss", None), best_metrics.get("best_val_acc", None)]],
    )
    wandb.log({"best_run_summary": summary_table})
    wandb.finish()
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Q2.2 dropout comparison trainer")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to Oxford-IIIT Pet dataset root")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per dropout setting")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio from trainval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--project", type=str, default="da6401_assignment2", help="W&B project name")
    parser.add_argument("--run-prefix", type=str, default="q22", help="Prefix for run names")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save best checkpoints")
    args = parser.parse_args()

    results: List[Dict[str, float]] = []
    for dropout_p in [0.0, 0.2, 0.5]:
        result = train_one_setting(
            data_dir=args.data_dir,
            dropout_p=dropout_p,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            project=args.project,
            run_prefix=args.run_prefix,
            save_dir=args.save_dir,
        )
        results.append(result)

    print("\n=== Q2.2 Summary ===")
    for res in results:
        print(res)


if __name__ == "__main__":
    main()
