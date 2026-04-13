import os
import copy
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.layers import CustomDropout


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 37
IMG_SIZE = 224


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG11EncoderQ21(nn.Module):
    """Standalone VGG11 encoder for Q2.1 so original files remain untouched."""

    def __init__(self, in_channels: int = 3, use_bn: bool = True):
        super().__init__()

        self.block1 = ConvBlock(in_channels, 64, use_bn)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = ConvBlock(64, 128, use_bn)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = nn.Sequential(
            ConvBlock(128, 256, use_bn),  # 3rd convolutional layer
            ConvBlock(256, 256, use_bn),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = nn.Sequential(
            ConvBlock(256, 512, use_bn),
            ConvBlock(512, 512, use_bn),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block5 = nn.Sequential(
            ConvBlock(512, 512, use_bn),
            ConvBlock(512, 512, use_bn),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, return_features: bool = False):
        features = {}

        x = self.block1(x)
        features["block1"] = x
        x = self.pool1(x)

        x = self.block2(x)
        features["block2"] = x
        x = self.pool2(x)

        x = self.block3(x)
        features["block3"] = x
        x = self.pool3(x)

        x = self.block4(x)
        features["block4"] = x
        x = self.pool4(x)

        x = self.block5(x)
        features["block5"] = x
        x = self.pool5(x)

        if return_features:
            return x, features
        return x


class VGG11ClassifierQ21(nn.Module):
    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn
        self.encoder = VGG11EncoderQ21(in_channels=in_channels, use_bn=use_bn)
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class ActivationCollector:
    def __init__(self):
        self.output = None

    def __call__(self, module, inp, out):
        self.output = out.detach().cpu()


def get_third_conv_layer(model: VGG11ClassifierQ21):
    # block3[0] is ConvBlock, block[0] inside it is Conv2d.
    return model.encoder.block3[0].block[0]


def build_loaders(data_dir: str, batch_size: int, val_split: float, seed: int, num_workers: int):
    full_dataset = OxfordIIITPetDataset(root_dir=data_dir, split="train", mask=False)

    val_len = int(len(full_dataset) * val_split)
    train_len = len(full_dataset) - val_len

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _, _ in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_single_run(data_dir: str, use_bn: bool, epochs: int, batch_size: int, lr: float,
                     dropout_p: float, val_split: float, seed: int, num_workers: int,
                     project: str, run_name: str, checkpoint_dir: str):
    set_seed(seed)
    train_loader, val_loader = build_loaders(data_dir, batch_size, val_split, seed, num_workers)

    model = VGG11ClassifierQ21(num_classes=NUM_CLASSES, dropout_p=dropout_p, use_bn=use_bn).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)

    run = wandb.init(
        project=project,
        name=run_name,
        config={
            "task": "q2.1_batchnorm_effect",
            "use_bn": use_bn,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "dropout_p": dropout_p,
            "val_split": val_split,
            "seed": seed,
        },
        reinit=True,
    )

    best_val_loss = float("inf")
    best_path = os.path.join(checkpoint_dir, f"{run_name}.pth")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels, _, _ in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"[{run_name}] Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    # Log activation histogram on the same validation sample.
    sample_images, sample_labels, _, _ = next(iter(val_loader))
    sample_images = sample_images[:1].to(DEVICE)
    sample_labels = sample_labels[:1]

    collector = ActivationCollector()
    handle = get_third_conv_layer(model).register_forward_hook(collector)
    model.eval()
    with torch.no_grad():
        sample_logits = model(sample_images)
    handle.remove()

    pred_label = sample_logits.argmax(dim=1).cpu().item()
    true_label = sample_labels.item()
    act = collector.output.flatten().numpy()

    wandb.log({
        "conv3_activation_hist": wandb.Histogram(act),
        "activation_mean": float(np.mean(act)),
        "activation_std": float(np.std(act)),
        "sample_true_label": true_label,
        "sample_pred_label": pred_label,
    })

    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["checkpoint_path"] = best_path
    run.finish()
    return {
        "best_val_loss": best_val_loss,
        "checkpoint_path": best_path,
        "true_label": true_label,
        "pred_label": pred_label,
        "activation_mean": float(np.mean(act)),
        "activation_std": float(np.std(act)),
    }


def lr_stability_scan(data_dir: str, learning_rates, epochs: int, batch_size: int,
                      dropout_p: float, val_split: float, seed: int, num_workers: int,
                      project: str, checkpoint_dir: str):
    results = []

    for use_bn in [False, True]:
        variant = "with_bn" if use_bn else "without_bn"
        for lr in learning_rates:
            set_seed(seed)
            train_loader, val_loader = build_loaders(data_dir, batch_size, val_split, seed, num_workers)
            model = VGG11ClassifierQ21(num_classes=NUM_CLASSES, dropout_p=dropout_p, use_bn=use_bn).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            stable = True
            final_train_loss = None
            final_val_loss = None

            run = wandb.init(
                project=project,
                name=f"lrscan_{variant}_lr_{lr}",
                config={
                    "task": "q2.1_lr_stability_scan",
                    "use_bn": use_bn,
                    "learning_rate": lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "dropout_p": dropout_p,
                    "val_split": val_split,
                    "seed": seed,
                },
                reinit=True,
            )

            try:
                for epoch in range(epochs):
                    model.train()
                    total_loss = 0.0
                    total_count = 0

                    for images, labels, _, _ in train_loader:
                        images = images.to(DEVICE)
                        labels = labels.to(DEVICE)

                        optimizer.zero_grad()
                        logits = model(images)
                        loss = criterion(logits, labels)

                        if not torch.isfinite(loss):
                            stable = False
                            raise RuntimeError("Non-finite loss encountered")

                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item() * images.size(0)
                        total_count += images.size(0)

                    final_train_loss = total_loss / max(total_count, 1)
                    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion)

                    if (not np.isfinite(final_train_loss)) or (not np.isfinite(final_val_loss)) or final_train_loss > 100.0:
                        stable = False
                        break

                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": final_train_loss,
                        "val_loss": final_val_loss,
                        "val_acc": final_val_acc,
                        "stable_so_far": int(stable),
                    })
            except Exception as exc:
                stable = False
                wandb.log({"stable_so_far": 0, "error": str(exc)})

            results.append([variant, float(lr), int(stable),
                            None if final_train_loss is None else float(final_train_loss),
                            None if final_val_loss is None else float(final_val_loss)])

            wandb.log({
                "lr_stability_table_partial": wandb.Table(
                    columns=["model", "learning_rate", "stable", "final_train_loss", "final_val_loss"],
                    data=results,
                )
            })
            run.finish()

    summary_run = wandb.init(project=project, name="q21_lr_stability_summary", reinit=True)
    wandb.log({
        "lr_stability_table": wandb.Table(
            columns=["model", "learning_rate", "stable", "final_train_loss", "final_val_loss"],
            data=results,
        )
    })
    summary_run.finish()

    stable_map = {"with_bn": [], "without_bn": []}
    for model_name, lr, stable, _, _ in results:
        if stable == 1:
            stable_map[model_name].append(lr)

    max_stable = {
        "with_bn": max(stable_map["with_bn"]) if stable_map["with_bn"] else None,
        "without_bn": max(stable_map["without_bn"]) if stable_map["without_bn"] else None,
    }
    return results, max_stable


def main():
    parser = argparse.ArgumentParser(description="Q2.1 BatchNorm comparison without touching original files")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to Oxford-IIIT Pet root directory")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout-p", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--project", type=str, default="da6401_assignment2")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--run-prefix", type=str, default="q21")
    parser.add_argument("--do-lr-scan", action="store_true")
    parser.add_argument("--lr-scan-epochs", type=int, default=3)
    parser.add_argument("--lr-values", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    result_no_bn = train_single_run(
        data_dir=args.data_dir,
        use_bn=False,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout_p=args.dropout_p,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        project=args.project,
        run_name=f"{args.run_prefix}_without_bn",
        checkpoint_dir=args.checkpoint_dir,
    )

    result_bn = train_single_run(
        data_dir=args.data_dir,
        use_bn=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout_p=args.dropout_p,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        project=args.project,
        run_name=f"{args.run_prefix}_with_bn",
        checkpoint_dir=args.checkpoint_dir,
    )

    print("\n===== Q2.1 Summary =====")
    print(f"Without BN -> best_val_loss={result_no_bn['best_val_loss']:.4f}, activation_std={result_no_bn['activation_std']:.4f}")
    print(f"With BN    -> best_val_loss={result_bn['best_val_loss']:.4f}, activation_std={result_bn['activation_std']:.4f}")

    if args.do_lr_scan:
        results, max_stable = lr_stability_scan(
            data_dir=args.data_dir,
            learning_rates=args.lr_values,
            epochs=args.lr_scan_epochs,
            batch_size=args.batch_size,
            dropout_p=args.dropout_p,
            val_split=args.val_split,
            seed=args.seed,
            num_workers=args.num_workers,
            project=args.project,
            checkpoint_dir=args.checkpoint_dir,
        )
        print("\n===== LR Stability Scan =====")
        for row in results:
            print(row)
        print(f"Max stable LR without BN: {max_stable['without_bn']}")
        print(f"Max stable LR with BN: {max_stable['with_bn']}")


if __name__ == "__main__":
    main()
