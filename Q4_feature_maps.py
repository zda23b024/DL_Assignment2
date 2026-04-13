import argparse
import math
import os
import random
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import random_split

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier

DOG_CLASS_IDS = set(range(25))  # Oxford-IIIT Pet: first 25 labels are dog breeds, last 12 are cat breeds


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, output):
        self.features = output.detach().cpu()

    def close(self):
        self.handle.remove()



def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()



def make_grid_image(feature_tensor: torch.Tensor, max_maps: int = 16, title: str = ""):
    # feature_tensor: [C, H, W]
    c = min(feature_tensor.shape[0], max_maps)
    cols = 4
    rows = math.ceil(c / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < c:
            fmap = feature_tensor[i].numpy()
            ax.imshow(fmap, cmap="gray")
            ax.set_title(f"Map {i}", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig



def find_dog_sample(dataset, preferred_index: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    if preferred_index is not None:
        _, label, _, _ = dataset[preferred_index]
        return dataset[preferred_index][0], int(label.item())

    for idx in range(len(dataset)):
        _, label, _, _ = dataset[idx]
        label_id = int(label.item())
        if label_id in DOG_CLASS_IDS:
            image_tensor, _, _, _ = dataset[idx]
            return image_tensor, label_id

    image_tensor, label, _, _ = dataset[0]
    return image_tensor, int(label.item())



def load_model(ckpt_path: Optional[str], device: torch.device, dropout_p: float):
    model = VGG11Classifier(num_classes=37, in_channels=3, dropout_p=dropout_p)
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        print("No valid checkpoint provided. Using current/random weights.")
    model.to(device)
    model.eval()
    return model



def main():
    parser = argparse.ArgumentParser(description="Q2.4 Feature map visualization helper")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset root dir")
    parser.add_argument("--classifier-ckpt", type=str, default="checkpoints/classifier.pth", help="Classifier checkpoint path")
    parser.add_argument("--sample-idx", type=int, default=None, help="Optional dataset index to visualize")
    parser.add_argument("--dropout-p", type=float, default=0.5, help="Classifier dropout used for model construction")
    parser.add_argument("--max-maps", type=int, default=16, help="How many feature maps to show from each layer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="q24_feature_maps")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="da6401_assignment2", name=args.run_name, config=vars(args))

    full_dataset = OxfordIIITPetDataset(root_dir=args.data_dir, split="train", mask=False)

    # keep same overall flow as your earlier helper scripts: create a train/val split from trainval
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_dataset, _ = random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    image_tensor, label_id = find_dog_sample(train_dataset, args.sample_idx)
    input_batch = image_tensor.unsqueeze(0).to(device)

    model = load_model(args.classifier_ckpt, device, args.dropout_p)

    first_conv = model.encoder.block1[0]
    last_conv = model.encoder.block5[3]

    hook_first = FeatureHook(first_conv)
    hook_last = FeatureHook(last_conv)

    with torch.no_grad():
        logits = model(input_batch)
        pred = int(torch.argmax(logits, dim=1).item())

    first_feats = hook_first.features.squeeze(0)
    last_feats = hook_last.features.squeeze(0)
    hook_first.close()
    hook_last.close()

    orig_img = denormalize(image_tensor)

    plt.figure(figsize=(4, 4))
    plt.imshow(orig_img)
    plt.axis("off")
    plt.title(f"Input dog image | GT label={label_id} | Pred={pred}")
    plt.tight_layout()
    wandb.log({"q24_input_image": wandb.Image(plt)})
    plt.close()

    fig1 = make_grid_image(first_feats, max_maps=args.max_maps, title="First convolutional layer feature maps")
    wandb.log({"q24_first_conv_feature_maps": wandb.Image(fig1)})
    plt.close(fig1)

    fig2 = make_grid_image(last_feats, max_maps=args.max_maps, title="Last convolutional layer before pooling feature maps")
    wandb.log({"q24_last_conv_feature_maps": wandb.Image(fig2)})
    plt.close(fig2)

    wandb.log({
        "q24_gt_label": label_id,
        "q24_pred_label": pred,
        "q24_first_conv_channels": int(first_feats.shape[0]),
        "q24_last_conv_channels": int(last_feats.shape[0]),
    })

    print("Logged input image and feature maps to W&B.")
    print("First conv layer used: model.encoder.block1[0]")
    print("Last conv before pooling used: model.encoder.block5[3]")

    wandb.finish()


if __name__ == "__main__":
    main()
