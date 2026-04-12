"""
Training entrypoint for DA6401 Assignment 2
Optimized for: 0.8+ Macro-Dice and 60%+ Acc@IoU=0.5
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# HELPERS
# =========================
def dice_loss(pred, target, num_classes=3):
    """Functional Dice Loss to maximize overlap score."""
    pred = torch.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + 1e-5) / (union + 1e-5)
    return 1 - dice.mean()

def freeze_encoder(model):
    """Freezes backbone to focus training on the task-specific heads."""
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("✅ Encoder Frozen")
    elif hasattr(model, 'features'):
        for param in model.features.parameters():
            param.requires_grad = False
        print("✅ Features Frozen")

# =========================
# LOCALIZER (Fixes 0.0% Acc@IoU)
# =========================
def train_localizer(data_dir, epochs=60, batch_size=32, lr=3e-5):
    wandb.init(project="da6401_assignment2", name="localizer_training")
    dataset = OxfordIIITPetDataset(root_dir=data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VGG11Localizer().to(DEVICE)
    freeze_encoder(model)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss()
    # Higher LR for head since encoder is frozen
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, _, boxes, _ in loader:
            images = images.to(DEVICE)
            # SCALE FIX: Maps [0,1] dataset boxes to [0,224] image space
            boxes = boxes.to(DEVICE).float() * 224.0

            preds = model(images)
            # Weighted loss to favor IoU overlap over simple distance
            loss = 0.5 * mse_loss(preds, boxes) + 5.0 * iou_loss(preds, boxes)

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        wandb.log({"localizer_loss": avg_loss})
        print(f"[Localizer] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/localizer.pth")
    wandb.finish()

# =========================
# SEGMENTATION (Fixes 0.22 Dice)
# =========================
def train_segmentation(data_dir, epochs=30, batch_size=16, lr=1e-4):
    wandb.init(project="da6401_assignment2", name="segmentation_training")
    dataset = OxfordIIITPetDataset(root_dir=data_dir, mask=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VGG11UNet(num_classes=3).to(DEVICE)
    freeze_encoder(model)

    # Weights: Background=1.0, Pet=5.0, Boundary=5.0 to combat background bias
    weights = torch.tensor([1.0, 5.0, 10.0]).to(DEVICE)
    ce_criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, _, _, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE).long()
            outputs = model(images)
            
            # Combine CE with Dice Loss to push Macro-Dice score above 0.8
            loss = ce_criterion(outputs, masks) + 2*dice_loss(outputs, masks)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        wandb.log({"segmentation_loss": avg_loss})
        print(f"[Segmentation] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/unet.pth")
    wandb.finish()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    DATA_DIR = "data"
    
    # You already have a perfect classifier score, so we skip it to save time.
    #print("🚀 Training Classifier...")
    #train_segmentation(DATA_DIR, epochs=50, batch_size=32, lr=1e-4)

    print("🚀 Training Localizer...")
    train_localizer(DATA_DIR, epochs=60, batch_size=32, lr=3e-5)
    
    #print("🚀 Training Segmentation...")
    #train_segmentation(DATA_DIR, epochs=30, batch_size=16, lr=1e-4)
