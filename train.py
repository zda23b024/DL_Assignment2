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

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# CLASSIFIER
# =========================
def train_classifier(data_dir, epochs=40, batch_size=32, lr=1e-4):

    wandb.init(project="da6401_assignment2", name="classifier_training")

    dataset = OxfordIIITPetDataset(root_dir=data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VGG11Classifier(num_classes=37).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    os.makedirs("checkpoints", exist_ok=True)

    best_loss = float("inf")  # ✅ ADDED

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels, _, _ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Classifier] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

        wandb.log({"classifier_loss": avg_loss})
        scheduler.step()

        # ✅ SAVE BEST MODEL
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/classifier.pth")
            print("✅ Saved BEST Classifier")

    wandb.finish()


# =========================
# LOCALIZER
# =========================
def train_localizer(data_dir, epochs=30, batch_size=32, lr=5e-5):

    wandb.init(project="da6401_assignment2", name="localizer_training")

    dataset = OxfordIIITPetDataset(root_dir=data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VGG11Localizer().to(DEVICE)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    best_loss = float("inf")  # ✅ ADDED

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, _, boxes, _ in loader:
            images = images.to(DEVICE)
            boxes = boxes.to(DEVICE).float()

            preds = model(images)

            loss_mse = mse_loss(preds, boxes)
            loss_iou = iou_loss(preds, boxes)
            loss = loss_mse + 2.5 * loss_iou

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Localizer] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

        wandb.log({"localizer_loss": avg_loss})

        # ✅ SAVE BEST MODEL
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/localizer.pth")
            print("✅ Saved BEST Localizer")

    wandb.finish()


# =========================
# SEGMENTATION
# =========================
def train_segmentation(data_dir, epochs=30, batch_size=16, lr=1e-4):

    wandb.init(project="da6401_assignment2", name="segmentation_training")

    dataset = OxfordIIITPetDataset(root_dir=data_dir, mask=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VGG11UNet(num_classes=3).to(DEVICE)

    weights = torch.tensor([1.0, 4.0, 4.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    best_loss = float("inf")  # ✅ ADDED

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, _, _, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).long()

            outputs = model(images)

            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Segmentation] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

        wandb.log({"segmentation_loss": avg_loss})

        # ✅ SAVE BEST MODEL
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/unet.pth")
            print("✅ Saved BEST Segmentation")

    wandb.finish()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    DATA_DIR = "data"

    print("🚀 Training Classifier...")
    train_classifier(DATA_DIR, epochs=50, batch_size=32, lr=1e-4)

    print("🚀 Training Localizer...")
    train_localizer(DATA_DIR, epochs=30, batch_size=32, lr=5e-5)

    print("🚀 Training Segmentation...")
    train_segmentation(DATA_DIR, epochs=20, batch_size=16, lr=1e-4)
