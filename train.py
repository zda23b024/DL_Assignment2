"""
Training entrypoint for DA6401 Assignment 2
Trains Classifier, Localizer, and Segmentation models with W&B logging
Saves checkpoints in `checkpoints/`
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

# Correct imports based on your folder structure
from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


def train_classifier(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train VGG11 classifier with W&B logging."""

    wandb.init(
        project="da6401_assignment2",
        name="classifier_training",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model": "VGG11Classifier"
        }
    )

    dataset = OxfordIIITPetDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VGG11Classifier(num_classes=37).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Optional: learning rate scheduler for better accuracy
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    os.makedirs("checkpoints", exist_ok=True)
    wandb.watch(model, log="all")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels, _, _ in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Classifier] Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "classifier_train_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        scheduler.step()  # Update LR if scheduler is used

    save_path = "checkpoints/classifier.pth"
    torch.save(model.state_dict(), save_path)
    artifact = wandb.Artifact("classifier_model", type="model")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    print(f"✅ Classifier saved at {save_path}")
    wandb.finish()


def train_localizer(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train Localizer model with W&B logging."""

    wandb.init(
        project="da6401_assignment2",
        name="localizer_training",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model": "LocalizerModel"
        }
    )

    dataset = OxfordIIITPetDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VGG11Localizer().to(device)
    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    wandb.watch(model, log="all")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, _, boxes, _ in dataloader:
            images, boxes = images.to(device), boxes.to(device).float()  # Ensure float

            preds = model(images)
            loss_mse = mse_loss(preds, boxes)
            loss_iou = iou_loss(preds, boxes)
            loss = loss_mse + loss_iou

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Localizer] Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "localizer_train_loss": avg_loss,
            "localizer_mse_loss": loss_mse.item(),
            "localizer_iou_loss": loss_iou.item(),
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

    save_path = "checkpoints/localizer.pth"
    torch.save(model.state_dict(), save_path)
    artifact = wandb.Artifact("localizer_model", type="model")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    print(f"✅ Localizer saved at {save_path}")
    wandb.finish()


def train_segmentation(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train segmentation UNet model with W&B logging."""

    wandb.init(
        project="da6401_assignment2",
        name="segmentation_training",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model": "UNet"
        }
    )

    dataset = OxfordIIITPetDataset(root_dir=data_dir, mask=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VGG11UNet(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    wandb.watch(model, log="all")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, _, _, masks in dataloader:
            images, masks = images.to(device), masks.to(device).long()  # ✅ convert to Long

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Segmentation] Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "segmentation_train_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

    save_path = "checkpoints/segmentation.pth"
    torch.save(model.state_dict(), save_path)
    artifact = wandb.Artifact("segmentation_model", type="model")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    print(f"✅ Segmentation model saved at {save_path}")
    wandb.finish()


if __name__ == "__main__":
    DATA_DIR = "data"

    # Train all models sequentially
    train_classifier(data_dir=DATA_DIR, epochs=10, batch_size=32)
    train_localizer(data_dir=DATA_DIR, epochs=10, batch_size=32)
    train_segmentation(data_dir=DATA_DIR, epochs=10, batch_size=16)
