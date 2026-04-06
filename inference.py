"""Inference and evaluation """

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

from data.pets_dataset import OxfordIIITPetDataset


def load_models(device):
    """Load all trained models"""

    # Classifier
    classifier = VGG11Classifier(num_classes=37).to(device)
    try:
        classifier.load_state_dict(
            torch.load("checkpoints/classifier.pth", map_location=device)
        )
        print("✅ Classifier loaded")
    except:
        print("⚠️ Could not load classifier")

    classifier.eval()

    # Localizer
    localizer = VGG11Localizer().to(device)
    try:
        localizer.load_state_dict(
            torch.load("checkpoints/localizer.pth", map_location=device)
        )
        print("✅ Localizer loaded")
    except:
        print("⚠️ Could not load localizer")

    localizer.eval()

    # Segmentation
    segmenter = VGG11UNet(num_classes=3).to(device)  # 3 classes: background, pet, border
    try:
        segmenter.load_state_dict(torch.load("checkpoints/unet.pth", map_location=device))
        print("✅ Segmentation model loaded")
    except Exception as e:
        print(f"❌ Segmentation load error: {e}")

    segmenter.eval()

    return classifier, localizer, segmenter


def predict(classifier, localizer, segmenter, image, device):
    """Run inference"""

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        cls_out = classifier(image)
        loc_out = localizer(image)
        seg_out = segmenter(image)

    pred_class = torch.argmax(cls_out, dim=1).item()
    bbox = loc_out.squeeze(0).cpu().numpy()
    seg_mask = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()

    return pred_class, bbox, seg_mask


def visualize(image, bbox, seg_mask):
    """Visualize results"""

    image = image.cpu().permute(1, 2, 0).numpy()

    # ✅ Unnormalize (fix visualization issue)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Image
    axs[0].imshow(image)
    axs[0].set_title("Image")

    # Bounding box
    axs[1].imshow(image)
    x_c, y_c, w, h = bbox
    x1 = x_c - w / 2
    y1 = y_c - h / 2

    rect = plt.Rectangle(
        (x1, y1), w, h,
        edgecolor='r',
        facecolor='none',
        linewidth=2
    )
    axs[1].add_patch(rect)
    axs[1].set_title("Bounding Box")

    # Segmentation
    axs[2].imshow(seg_mask, cmap="gray")
    axs[2].set_title("Segmentation")

    for ax in axs:
        ax.axis("off")

    plt.show()


def run_inference(data_dir="data", index=0):
    """Run full pipeline"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset (mask=True needed for segmentation)
    dataset = OxfordIIITPetDataset(root_dir=data_dir, mask=True)

    image, label, bbox_gt, mask_gt = dataset[index]

    # Load models
    classifier, localizer, segmenter = load_models(device)

    # Predict
    pred_class, pred_bbox, pred_mask = predict(
        classifier, localizer, segmenter, image, device
    )

    print(f"Predicted Class: {pred_class}")
    print(f"Predicted BBox: {pred_bbox}")

    # Visualize
    visualize(image, pred_bbox, pred_mask)


if __name__ == "__main__":
    run_inference()