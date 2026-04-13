"""
Q2.8 helper: Meta-analysis and reflection for the unified multi-task pipeline.

This script does NOT modify the original project code.
It is meant to help assemble the evidence needed for Question 2.8 by:
1) Pulling metrics from completed W&B runs for Q2.1–Q2.7
2) Producing clean comparison plots as image files
3) Optionally creating a markdown summary scaffold you can paste into your W&B report

It expects that you already ran the earlier helper scripts, or at least have
relevant runs in the same W&B project.

Usage:
    python train_q28_meta_analysis.py \
        --entity YOUR_WANDB_ENTITY \
        --project da6401_assignment2

Optional:
    python train_q28_meta_analysis.py \
        --entity YOUR_WANDB_ENTITY \
        --project da6401_assignment2 \
        --output-dir q28_outputs \
        --run-prefix q28

Requirements:
    pip install wandb matplotlib pandas
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import wandb


Q28_PROMPT = """
Question 2.8 asks for:
- comprehensive overlaid metric plots across training history
- retrospective architectural reflection
- comments on:
  1) dropout + batch norm placement
  2) frozen vs fine-tuned shared backbone
  3) segmentation loss effectiveness
"""


DEFAULT_GROUPS = {
    "q21_batchnorm": ["q21_without_bn", "q21_with_bn"],
    "q22_dropout": ["q22_dropout_0.0", "q22_dropout_0.2", "q22_dropout_0.5"],
    "q23_transfer": [
        "q23_transfer_strict_feature_extractor",
        "q23_transfer_partial_finetune",
        "q23_transfer_full_finetune",
    ],
    "q26_segmentation": ["q26_segmentation_metrics"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Q2.8 meta-analysis helper")
    parser.add_argument("--entity", type=str, required=True, help="Your W&B username or team")
    parser.add_argument("--project", type=str, default="da6401_assignment2", help="W&B project name")
    parser.add_argument("--output-dir", type=str, default="q28_outputs", help="Where plots and markdown are saved")
    parser.add_argument("--run-prefix", type=str, default="q28", help="Prefix for generated artifact names")
    return parser.parse_args()


def safe_get_history(run, keys: List[str]) -> Optional[pd.DataFrame]:
    try:
        hist = run.history(keys=keys, pandas=True)
        if hist is None or len(hist) == 0:
            return None
        return hist
    except Exception:
        return None


def find_runs_by_exact_name(api, entity: str, project: str, names: List[str]):
    results = {}
    runs = api.runs(f"{entity}/{project}")
    for name in names:
        results[name] = None
        for run in runs:
            if run.name == name:
                results[name] = run
                break
    return results


def pick_epoch_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ["epoch", "_step"]:
        if cand in df.columns:
            return cand
    return None


def save_lineplot(dfs: Dict[str, pd.DataFrame], metric: str, title: str, outpath: Path):
    plt.figure(figsize=(8, 5))
    plotted = False

    for run_name, df in dfs.items():
        if df is None or metric not in df.columns:
            continue
        xcol = pick_epoch_column(df)
        if xcol is None:
            continue
        sub = df[[xcol, metric]].dropna()
        if len(sub) == 0:
            continue
        plt.plot(sub[xcol], sub[metric], label=run_name)
        plotted = True

    plt.title(title)
    plt.xlabel("Epoch / Step")
    plt.ylabel(metric)
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return plotted


def collect_group_histories(api, entity: str, project: str, group_names: List[str], keys: List[str]):
    found = find_runs_by_exact_name(api, entity, project, group_names)
    histories = {}
    for run_name, run in found.items():
        histories[run_name] = safe_get_history(run, keys) if run is not None else None
    return histories, found


def write_markdown_summary(outdir: Path, generated_plots: Dict[str, List[Path]], found_runs: Dict[str, Dict[str, object]]):
    md = []
    md.append("# Q2.8 Meta-Analysis Scaffold\n")
    md.append("Use the generated plots below inside your W&B report.\n")
    md.append("## Available Runs\n")
    for group, mapping in found_runs.items():
        md.append(f"### {group}\n")
        for name, run in mapping.items():
            status = "FOUND" if run is not None else "MISSING"
            md.append(f"- {name}: {status}\n")
        md.append("")
    md.append("## Generated Plot Files\n")
    for group, paths in generated_plots.items():
        md.append(f"### {group}\n")
        for p in paths:
            md.append(f"- `{p.name}`\n")
        md.append("")
    md.append("## Reflection Outline\n")
    md.append("### 1. Architectural Reasoning (Dropout + BatchNorm)\n")
    md.append("- Compare Q2.1 and Q2.2 curves.\n")
    md.append("- Discuss whether BN improved optimization stability and whether moderate dropout reduced overfitting.\n")
    md.append("- Argue why this placement helped the final unified model.\n")
    md.append("")
    md.append("### 2. Encoder Adaptation (Frozen vs Fine-Tuned)\n")
    md.append("- Use Q2.3 Dice / loss / epoch-time plots.\n")
    md.append("- State whether strict freezing under-adapted, and whether partial or full fine-tuning gave the best trade-off.\n")
    md.append("- Discuss task interference in the shared backbone.\n")
    md.append("")
    md.append("### 3. Segmentation Loss Evaluation\n")
    md.append("- Use Q2.6 Dice vs pixel-accuracy plots.\n")
    md.append("- Explain why Dice is more informative for imbalanced foreground-background segmentation.\n")
    md.append("- Reflect on whether your chosen segmentation loss aligned with visual mask quality.\n")
    md.append("")
    md.append("### 4. Unified Pipeline Reflection\n")
    md.append("- Link the classification regularization choices, localization backbone adaptation, and segmentation loss to the final pipeline behavior.\n")
    md.append("- Mention strengths, weaknesses, and likely causes of failure on wild images from Q2.7.\n")

    md_path = outdir / "q28_meta_analysis_scaffold.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return md_path


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()

    generated_plots = {}
    found_all = {}

    # Q2.1 plots
    q21_keys = ["epoch", "_step", "train_loss", "val_loss", "val_acc"]
    q21_hist, q21_found = collect_group_histories(api, args.entity, args.project, DEFAULT_GROUPS["q21_batchnorm"], q21_keys)
    found_all["q21_batchnorm"] = q21_found
    q21_paths = []
    for metric in ["train_loss", "val_loss", "val_acc"]:
        p = outdir / f"{args.run_prefix}_q21_{metric}.png"
        if save_lineplot(q21_hist, metric, f"Q2.1 BatchNorm Comparison - {metric}", p):
            q21_paths.append(p)
    generated_plots["q21_batchnorm"] = q21_paths

    # Q2.2 plots
    q22_keys = ["epoch", "_step", "train_loss", "val_loss", "train_acc", "val_acc", "generalization_gap"]
    q22_hist, q22_found = collect_group_histories(api, args.entity, args.project, DEFAULT_GROUPS["q22_dropout"], q22_keys)
    found_all["q22_dropout"] = q22_found
    q22_paths = []
    for metric in ["train_loss", "val_loss", "generalization_gap", "val_acc"]:
        p = outdir / f"{args.run_prefix}_q22_{metric}.png"
        if save_lineplot(q22_hist, metric, f"Q2.2 Dropout Comparison - {metric}", p):
            q22_paths.append(p)
    generated_plots["q22_dropout"] = q22_paths

    # Q2.3 plots
    q23_keys = ["epoch", "_step", "train_loss", "val_loss", "train_dice", "val_dice", "train_pixel_acc", "val_pixel_acc", "epoch_time_sec"]
    q23_hist, q23_found = collect_group_histories(api, args.entity, args.project, DEFAULT_GROUPS["q23_transfer"], q23_keys)
    found_all["q23_transfer"] = q23_found
    q23_paths = []
    for metric in ["val_loss", "val_dice", "val_pixel_acc", "epoch_time_sec"]:
        p = outdir / f"{args.run_prefix}_q23_{metric}.png"
        if save_lineplot(q23_hist, metric, f"Q2.3 Transfer Strategy Comparison - {metric}", p):
            q23_paths.append(p)
    generated_plots["q23_transfer"] = q23_paths

    # Q2.6 plots
    q26_keys = ["epoch", "_step", "train_loss", "val_loss", "train_dice", "val_dice", "train_pixel_acc", "val_pixel_acc"]
    q26_hist, q26_found = collect_group_histories(api, args.entity, args.project, DEFAULT_GROUPS["q26_segmentation"], q26_keys)
    found_all["q26_segmentation"] = q26_found
    q26_paths = []
    for metric in ["val_loss", "val_dice", "val_pixel_acc"]:
        p = outdir / f"{args.run_prefix}_q26_{metric}.png"
        if save_lineplot(q26_hist, metric, f"Q2.6 Segmentation Metrics - {metric}", p):
            q26_paths.append(p)
    generated_plots["q26_segmentation"] = q26_paths

    md_path = write_markdown_summary(outdir, generated_plots, found_all)

    print("Q2.8 helper finished.")
    print(f"Output directory: {outdir.resolve()}")
    print("Generated files:")
    for group, paths in generated_plots.items():
        for p in paths:
            print(f" - {p.resolve()}")
    print(f" - {md_path.resolve()}")


if __name__ == "__main__":
    main()
