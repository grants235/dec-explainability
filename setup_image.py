#!/usr/bin/env python3
"""
setup_image.py
==============
Stage-1 setup for the teleological-xai image pipeline.

Covers everything up to — but NOT including — the novel teleological methods
(Purposive Saliency, Means-End Decomposition).

Steps
-----
1. Download & extract CUB-200-2011 into data/CUB_200_2011/
2. Train ResNet-50 on CUB-200-2011  (skipped if checkpoint already exists)
3. Performance evaluation  (top-1, top-5, per-class, confusion matrix)
4. Baseline explainability evaluation  (GradCAM, IG, SHAP)
   - Deletion AUC, Insertion AUC, PBPA, timing
5. Write results/image/  with CSV + LaTeX tables, PNG visualisations

Usage
-----
    .venv/bin/python setup_image.py                        # full run
    .venv/bin/python setup_image.py --skip-training        # skip if ckpt exists
    .venv/bin/python setup_image.py --epochs 10            # quick smoke test
    .venv/bin/python setup_image.py --n-eval 50            # evaluate 50 images
    .venv/bin/python setup_image.py --device cpu           # force CPU
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Paths (all relative to the directory this script lives in)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR / "data" / "CUB_200_2011"
CKPT_DIR     = SCRIPT_DIR / "models" / "image" / "checkpoint"
RESULTS_DIR  = SCRIPT_DIR / "results" / "image"

CUB_URL      = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
CUB_TGZ      = SCRIPT_DIR / "data" / "CUB_200_2011.tgz"
BEST_CKPT    = CKPT_DIR / "resnet50_cub200_best.pth"

IMAGE_SIZE   = 448
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 200


# ===========================================================================
# 0. Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CUB-200 image pipeline: download → train → eval → baselines"
    )
    p.add_argument("--data-dir", type=Path, default=DATA_DIR,
                   help="Path to CUB_200_2011 root (default: data/CUB_200_2011)")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                   help="Directory to write all outputs (default: results/image)")
    p.add_argument("--checkpoint", type=Path, default=BEST_CKPT,
                   help="Path to trained model checkpoint (best.pth)")
    p.add_argument("--epochs", type=int, default=60,
                   help="Training epochs (default: 60)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Training batch size (default: 16)")
    p.add_argument("--n-eval", type=int, default=200,
                   help="Number of test images for explainability eval (default: 200)")
    p.add_argument("--n-shap-bg", type=int, default=200,
                   help="Background samples for SHAP (default: 200)")
    p.add_argument("--device", type=str, default=None,
                   help="Device: 'cuda', 'mps', or 'cpu' (auto-detected if omitted)")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training if a checkpoint already exists")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip dataset download (assume data-dir already populated)")
    return p.parse_args()


# ===========================================================================
# 1. Dataset download
# ===========================================================================

def download_cub(data_dir: Path, skip: bool) -> None:
    if skip or (data_dir / "images.txt").exists():
        print(f"[1/4] CUB-200-2011 already present at {data_dir}  (skip download)")
        return

    print("[1/4] Downloading CUB-200-2011 …")
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    if not CUB_TGZ.exists():
        print(f"      Fetching {CUB_URL}")
        def _progress(block, bsize, total):
            if total > 0:
                pct = min(100, block * bsize * 100 // total)
                print(f"\r      {pct:3d}%", end="", flush=True)
        urllib.request.urlretrieve(CUB_URL, CUB_TGZ, reporthook=_progress)
        print()
    else:
        print(f"      Archive already at {CUB_TGZ}")

    print(f"      Extracting to {data_dir.parent} …")
    import tarfile
    with tarfile.open(CUB_TGZ) as tf:
        tf.extractall(data_dir.parent)

    if not (data_dir / "images.txt").exists():
        raise RuntimeError(
            f"Extraction finished but {data_dir / 'images.txt'} not found. "
            "Check the archive structure."
        )
    print(f"      Dataset ready at {data_dir}")


# ===========================================================================
# 2. Model training
# ===========================================================================

def train_model(args: argparse.Namespace) -> None:
    if args.skip_training and args.checkpoint.exists():
        print(f"[2/4] Checkpoint found at {args.checkpoint}  (skip training)")
        return
    if not args.skip_training and args.checkpoint.exists():
        print(f"[2/4] Checkpoint found at {args.checkpoint}  (re-using existing)")
        return

    print(f"[2/4] Training ResNet-50 for {args.epochs} epochs …")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "models" / "image" / "train_classifier.py"),
        "--data-dir",    str(args.data_dir),
        "--save-dir",    str(SCRIPT_DIR),
        "--num-epochs",  str(args.epochs),
        "--batch-size",  str(args.batch_size),
    ]
    if args.device:
        cmd += ["--device", args.device]

    print("      Running:", " ".join(cmd))
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        raise RuntimeError("Training script exited with non-zero status.")

    if not args.checkpoint.exists():
        raise RuntimeError(
            f"Training finished but expected checkpoint not found: {args.checkpoint}"
        )
    print(f"      Checkpoint saved at {args.checkpoint}")


# ===========================================================================
# Helper: lazy torch import (avoids paying startup cost before steps 1-2)
# ===========================================================================

def _load_torch_deps():
    """Import heavy torch / project modules only when needed."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms, models
    from torch.utils.data import DataLoader

    # Add project root to path so that project imports work
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    return torch, F, transforms, models, DataLoader


def _auto_device(requested: Optional[str]) -> "torch.device":
    import torch
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model_and_data(args: argparse.Namespace):
    """Load the trained ResNet-50 and the CUB test DataLoader."""
    torch, F, transforms, models, DataLoader = _load_torch_deps()
    device = _auto_device(args.device)

    from models.image.train_classifier import Cub2011Dataset

    val_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_ds = Cub2011Dataset(
        root_dir=str(args.data_dir),
        transform=val_transform,
        train=False,
        use_bbox=False,
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=2, pin_memory=(device.type == "cuda"))

    # Build model skeleton and load weights
    model = models.resnet50(weights=None)
    import torch.nn as nn
    model.fc = nn.Linear(2048, NUM_CLASSES)
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, test_ds, test_loader, device


# ===========================================================================
# 3. Performance evaluation
# ===========================================================================

def evaluate_performance(args: argparse.Namespace) -> Dict:
    import torch
    print("[3/4] Performance evaluation …")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    model, test_ds, test_loader, device = _load_model_and_data(args)

    top1_correct = 0
    top5_correct = 0
    total = 0
    per_class_correct = np.zeros(NUM_CLASSES, dtype=int)
    per_class_total   = np.zeros(NUM_CLASSES, dtype=int)
    conf_matrix       = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    from tqdm import tqdm
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="      Eval", ncols=80, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)                          # (B, 200)
            probs  = torch.softmax(logits, dim=1)

            # Top-1
            pred1 = logits.argmax(dim=1)
            top1_correct += (pred1 == labels).sum().item()

            # Top-5
            top5 = logits.topk(5, dim=1).indices
            for i, lbl in enumerate(labels):
                if lbl.item() in top5[i].tolist():
                    top5_correct += 1
                per_class_correct[lbl.item()] += int(pred1[i].item() == lbl.item())
                per_class_total[lbl.item()]   += 1
                conf_matrix[lbl.item(), pred1[i].item()] += 1

            total += labels.size(0)

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    metrics = {
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "n_test_images": total,
        "n_classes": NUM_CLASSES,
    }
    print(f"      Top-1 accuracy: {top1_acc*100:.2f}%")
    print(f"      Top-5 accuracy: {top5_acc*100:.2f}%")

    # --- per-class CSV ---
    class_names = test_ds.class_names        # {int -> str}
    per_class_acc = np.where(
        per_class_total > 0,
        per_class_correct / np.maximum(per_class_total, 1),
        np.nan,
    )
    per_class_path = args.results_dir / "per_class_accuracy.csv"
    with open(per_class_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "n_total", "n_correct", "accuracy"])
        for i in range(NUM_CLASSES):
            w.writerow([
                i,
                class_names.get(i, f"class_{i}"),
                int(per_class_total[i]),
                int(per_class_correct[i]),
                f"{per_class_acc[i]:.4f}" if not np.isnan(per_class_acc[i]) else "nan",
            ])

    # --- confusion matrix heat-map (top-20 most confused classes) ---
    _plot_top_confused(conf_matrix, class_names, args.results_dir)

    # --- summary JSON ---
    perf_path = args.results_dir / "performance_metrics.json"
    with open(perf_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Saved → {perf_path}")

    return metrics


def _plot_top_confused(conf_matrix: np.ndarray, class_names: Dict[int, str],
                       out_dir: Path) -> None:
    """Plot a 20×20 sub-matrix of the most frequently confused class pairs."""
    # Off-diagonal confusion counts
    off_diag = conf_matrix.copy()
    np.fill_diagonal(off_diag, 0)
    row_totals = off_diag.sum(axis=1)
    top20 = np.argsort(row_totals)[-20:][::-1]
    sub = conf_matrix[np.ix_(top20, top20)].astype(float)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sub, cmap="Blues", interpolation="nearest")
    labels = [class_names.get(i, str(i)).replace("_", " ")[:20] for i in top20]
    ax.set_xticks(range(20)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(20)); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Confusion Matrix — Top-20 Most Confused Classes")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    p = out_dir / "confusion_matrix_top20.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {p}")


# ===========================================================================
# 4. Baseline explainability evaluation
# ===========================================================================

def evaluate_baselines(args: argparse.Namespace, perf_metrics: Dict) -> None:
    import torch
    print(f"[4/4] Baseline explainability evaluation (n={args.n_eval} images) …")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    qual_dir = args.results_dir / "qualitative"
    qual_dir.mkdir(exist_ok=True)

    model, test_ds, test_loader, device = _load_model_and_data(args)

    # --- import explainers ---
    from methods.baselines.gradcam           import GradCAMExplainer
    from methods.baselines.integrated_gradients import IGExplainer
    from methods.baselines.shap_explainer    import SHAPExplainer
    from evaluation.metrics                  import (
        compute_deletion_auc,
        compute_insertion_auc,
        compute_pbpa,
        compute_diagnostic_parts,
        CUB_PART_NAMES,
    )
    from torchvision import transforms

    # --- build explainers ---
    gradcam_exp = GradCAMExplainer(model, device=str(device))
    ig_exp      = IGExplainer(model, device=str(device))
    shap_exp    = SHAPExplainer(
        model,
        background_dataset=test_ds,
        n_background=args.n_shap_bg,
        device=str(device),
    )

    # --- load annotation caches ---
    part_locs     = _load_pickle(CKPT_DIR / "part_locations.pkl")     # may be None
    img_attrs     = _load_pickle(CKPT_DIR / "image_attributes.pkl")   # may be None
    cls_attr_mat  = _load_npz(CKPT_DIR / "class_attribute_matrix.npz")  # may be None

    # dataset mean for deletion baseline (un-normalize then re-normalize)
    dataset_mean_tensor = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)   # black baseline

    # denormalise for display
    inv_mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    inv_std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    methods = {
        "GradCAM":             gradcam_exp,
        "IntegratedGradients": ig_exp,
        "SHAP":                shap_exp,
    }

    # rows: one per (method, image)
    rows: List[Dict] = []

    n_done      = 0
    n_visualise = min(20, args.n_eval)   # save PNG for first 20

    from tqdm import tqdm
    pbar = tqdm(total=args.n_eval, desc="      Eval", ncols=80)

    for img_idx in range(len(test_ds)):
        if n_done >= args.n_eval:
            break

        img_tensor, label = test_ds[img_idx]        # (3, H, W), int
        x = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_label = int(probs.argmax())

        # --- img_id for annotation lookup ---
        # test_ds.samples[img_idx] = (path, label, bbox)
        img_id = _extract_img_id(test_ds, img_idx)

        # --- part keypoints for PBPA ---
        part_keypoints = _get_part_keypoints(part_locs, img_id, IMAGE_SIZE)

        # --- diagnostic parts using class-level attributes ---
        diagnostic_parts: List[str] = []
        if cls_attr_mat is not None:
            try:
                conf_competitors = _top_k_competitors(probs, label, k=3)
                for comp in conf_competitors:
                    dp = compute_diagnostic_parts(
                        cls_attr_mat, label, comp, CUB_PART_NAMES
                    )
                    diagnostic_parts.extend(dp)
                diagnostic_parts = list(set(diagnostic_parts))
            except Exception:
                pass

        for method_name, explainer in methods.items():
            t0 = time.perf_counter()
            try:
                saliency = _compute_saliency(explainer, x, label, method_name)
            except Exception as e:
                print(f"\n      WARNING: {method_name} failed on img {img_idx}: {e}")
                continue
            elapsed = time.perf_counter() - t0

            # --- metrics ---
            del_auc = float("nan")
            ins_auc = float("nan")
            pbpa    = float("nan")
            try:
                del_auc = compute_deletion_auc(
                    model, x, saliency, label, device=str(device), n_steps=21
                )
            except Exception:
                pass
            try:
                ins_auc = compute_insertion_auc(
                    model, x, saliency, label, device=str(device), n_steps=21
                )
            except Exception:
                pass
            if part_keypoints and diagnostic_parts:
                try:
                    pbpa = compute_pbpa(saliency, part_keypoints, diagnostic_parts)
                except Exception:
                    pass

            rows.append({
                "img_idx":         img_idx,
                "true_label":      label,
                "pred_label":      pred_label,
                "correct":         int(pred_label == label),
                "method":          method_name,
                "deletion_auc":    round(del_auc, 4) if not np.isnan(del_auc) else None,
                "insertion_auc":   round(ins_auc, 4) if not np.isnan(ins_auc) else None,
                "pbpa":            round(pbpa, 4)    if not np.isnan(pbpa)    else None,
                "time_sec":        round(elapsed, 3),
            })

            # --- qualitative visualisation ---
            if n_done < n_visualise:
                try:
                    _save_qual_vis(
                        x, saliency, method_name, label,
                        test_ds.class_names, img_idx, qual_dir,
                        inv_mean, inv_std,
                    )
                except Exception:
                    pass

        n_done += 1
        pbar.update(1)

    pbar.close()

    # --- write CSV ---
    csv_path = args.results_dir / "baseline_explainability.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"      Saved → {csv_path}")

    # --- aggregate summary ---
    summary = _aggregate_baseline_summary(rows)
    summary_path = args.results_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      Saved → {summary_path}")

    # --- LaTeX table ---
    _write_latex_table(summary, args.results_dir / "baseline_summary.tex",
                       perf_metrics)

    # --- bar chart ---
    _plot_summary_chart(summary, args.results_dir / "baseline_summary.png")
    print("      Done with baseline evaluation.")


# ---------------------------------------------------------------------------
# Helpers for baseline eval
# ---------------------------------------------------------------------------

def _load_pickle(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _load_npz(path: Path):
    if path.exists():
        data = np.load(str(path))
        # class_attribute_matrix.npz stores key 'arr' or 'data'
        for key in ["arr_0", "data", "class_attribute_matrix"]:
            if key in data:
                return data[key]
        # fall back: first array
        keys = list(data.files)
        return data[keys[0]] if keys else None
    return None


def _extract_img_id(dataset, idx: int) -> Optional[int]:
    """Extract the raw CUB image_id from the dataset samples list."""
    try:
        # samples = [(img_path, label, bbox), ...]
        # img_id can be inferred from path name, e.g. '001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
        # The dataset also stores a .data DataFrame with img_id info
        if hasattr(dataset, "data"):
            import pandas as pd
            row = dataset.data.iloc[idx]
            return int(row.name) if hasattr(row, "name") else int(getattr(row, "img_id", idx + 1))
        return idx + 1
    except Exception:
        return None


def _get_part_keypoints(
    part_locs: Optional[Dict],
    img_id: Optional[int],
    image_size: int,
) -> Dict[str, Optional[Tuple[int, int]]]:
    """Return {part_name: (row, col)} or empty dict."""
    from evaluation.metrics import CUB_PART_NAMES
    if part_locs is None or img_id is None:
        return {}
    entry = part_locs.get(img_id, {})
    keypoints: Dict[str, Optional[Tuple[int, int]]] = {}
    # part_locs[img_id][part_id] = (x, y, visible)   (x=col, y=row, 1-indexed)
    # part_id 1-15 mapping:
    PART_ID_TO_NAME = {i + 1: name for i, name in enumerate(CUB_PART_NAMES)}
    for part_id, (x, y, visible) in entry.items():
        name = PART_ID_TO_NAME.get(part_id)
        if name is None:
            continue
        if int(visible):
            # CUB coords are pixel coordinates on the original image;
            # our images are resized to image_size — approximate scaling
            keypoints[name] = (int(y), int(x))   # (row, col)
        else:
            keypoints[name] = None
    return keypoints


def _top_k_competitors(probs: np.ndarray, true_class: int, k: int = 3) -> List[int]:
    sorted_idx = np.argsort(probs)[::-1]
    return [int(i) for i in sorted_idx if i != true_class][:k]


def _compute_saliency(explainer, x, label: int, method_name: str) -> np.ndarray:
    """Dispatch to the right explainer method."""
    if method_name == "GradCAM":
        return explainer.compute(x, label)
    elif method_name == "IntegratedGradients":
        return explainer.compute(x, label)
    elif method_name == "SHAP":
        return explainer.compute(x, label)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def _save_qual_vis(
    x, saliency: np.ndarray, method_name: str, label: int,
    class_names: Dict, img_idx: int, out_dir: Path,
    inv_mean, inv_std,
) -> None:
    import torch
    img_np = (x[0].cpu() * inv_std + inv_mean).permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np)
    axes[0].set_title(f"Image\n{class_names.get(label, label)}", fontsize=8)
    axes[0].axis("off")

    axes[1].imshow(saliency, cmap="inferno")
    axes[1].set_title(f"{method_name}\nSaliency Map", fontsize=8)
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(saliency, cmap="inferno", alpha=0.5)
    axes[2].set_title("Overlay", fontsize=8)
    axes[2].axis("off")

    plt.tight_layout()
    fname = out_dir / f"img{img_idx:04d}_{method_name}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _aggregate_baseline_summary(rows: List[Dict]) -> Dict:
    from collections import defaultdict
    buckets: Dict[str, Dict[str, List]] = defaultdict(lambda: {
        "deletion_auc": [], "insertion_auc": [], "pbpa": [], "time_sec": []
    })
    for r in rows:
        m = r["method"]
        for metric in ("deletion_auc", "insertion_auc", "pbpa", "time_sec"):
            v = r.get(metric)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                buckets[m][metric].append(v)

    summary = {}
    for method, data in buckets.items():
        summary[method] = {}
        for metric, vals in data.items():
            if vals:
                summary[method][metric] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std":  round(float(np.std(vals)),  4),
                    "n":    len(vals),
                }
    return summary


def _write_latex_table(summary: Dict, path: Path, perf: Dict) -> None:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Baseline XAI methods on CUB-200-2011 "
        + rf"(Top-1 acc = {perf.get('top1_accuracy', 0)*100:.1f}" + r"\%)}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Del AUC}$\downarrow$ "
        r"& \textbf{Ins AUC}$\uparrow$ & \textbf{PBPA}$\uparrow$ "
        r"& \textbf{Time (s)}\\",
        r"\midrule",
    ]
    for method, data in summary.items():
        def _fmt(key):
            d = data.get(key, {})
            if not d:
                return "---"
            return rf"{d['mean']:.3f} \pm {d['std']:.3f}"
        lines.append(
            rf"{method} & {_fmt('deletion_auc')} & {_fmt('insertion_auc')} "
            rf"& {_fmt('pbpa')} & {_fmt('time_sec')} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))
    print(f"      Saved → {path}")


def _plot_summary_chart(summary: Dict, path: Path) -> None:
    metrics = ["deletion_auc", "insertion_auc", "pbpa"]
    labels  = ["Del AUC ↓", "Ins AUC ↑", "PBPA ↑"]
    methods = list(summary.keys())
    n_methods = len(methods)
    if n_methods == 0:
        return

    x = np.arange(len(metrics))
    width = 0.8 / n_methods
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (method, color) in enumerate(zip(methods, colors)):
        means = []
        errs  = []
        for m in metrics:
            d = summary[method].get(m, {})
            means.append(d.get("mean", 0))
            errs.append(d.get("std", 0))
        offsets = x + (i - n_methods / 2 + 0.5) * width
        ax.bar(offsets, means, width, label=method, color=color, yerr=errs,
               capsize=4, error_kw={"elinewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("Baseline Explainability Methods — CUB-200-2011")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Teleological XAI — Image Pipeline Setup")
    print("=" * 60)
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Eval images: {args.n_eval}")
    print("=" * 60)

    # Step 1 — download
    download_cub(args.data_dir, skip=args.skip_download)

    # Step 2 — train
    train_model(args)

    # Ensure project root on path for subsequent imports
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    # Step 3 — performance evaluation
    perf = evaluate_performance(args)

    # Step 4 — baseline explainability
    evaluate_baselines(args, perf)

    print()
    print("=" * 60)
    print("  Setup complete!  Results written to:")
    print(f"    {args.results_dir}/")
    print()
    print("  Key outputs:")
    print(f"    performance_metrics.json    — top-1/top-5 accuracy")
    print(f"    per_class_accuracy.csv      — per-species accuracy")
    print(f"    confusion_matrix_top20.png  — top-20 confusion heat-map")
    print(f"    baseline_explainability.csv — per-image Del/Ins AUC, PBPA")
    print(f"    baseline_summary.json       — aggregated baseline metrics")
    print(f"    baseline_summary.tex        — LaTeX results table")
    print(f"    baseline_summary.png        — bar chart")
    print(f"    qualitative/                — 20 sample visualisations")
    print()
    print("  Next step — run teleological methods:")
    print("    .venv/bin/python experiments/run_image_experiments.py \\")
    print(f"      --config configs/image_config.yaml \\")
    print(f"      --data-dir {args.data_dir} \\")
    print(f"      --checkpoint {args.checkpoint} \\")
    print(f"      --results-dir {args.results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
