"""
Generate paper-quality figures for the teleological XAI paper.

Figures produced
----------------
1. ``comparison_bar.pdf``       – all methods × all image metrics
2. ``qualitative_grid.pdf``     – 4×5 grid of curated qualitative images
3. ``teleological_flow.pdf``    – Means-End layer-wise discriminability flow
4. ``subgoal_timeline.pdf``     – Sub-goal timeline plot for RL trajectories

Usage
-----
    python experiments/analysis/plot_results.py \\
        --image-results  results/image/ \\
        --rl-results     results/rl/ \\
        --output-dir     results/figures/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Colour palette (colour-blind friendly)
# ---------------------------------------------------------------------------
METHOD_COLORS: Dict[str, str] = {
    "purposive": "#2196F3",  # blue
    "ig":        "#FF9800",  # orange
    "gradcam":   "#4CAF50",  # green
    "shap":      "#9C27B0",  # purple
}
METHOD_LABELS: Dict[str, str] = {
    "purposive": "Purposive (ours)",
    "ig":        "Integrated Gradients",
    "gradcam":   "GradCAM",
    "shap":      "GradientSHAP",
}

METRIC_DISPLAY: Dict[str, str] = {
    "pbpa_mean":                     "PBPA",
    "deletion_auc_mean":             "Del AUC",
    "insertion_auc_mean":            "Ins AUC",
    "purposive_specificity_mean":    "Spec (PS)",
}


# ---------------------------------------------------------------------------
# Figure 1: Comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison_bar(
    summary: Dict[str, Dict[str, float]],
    output_path: str,
    methods: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> None:
    """
    Grouped bar chart comparing all methods on all image metrics.

    Parameters
    ----------
    summary : dict
        Nested dict: method -> metric_name -> value  (from image_summary.json).
    output_path : str
        Destination PDF/PNG.
    """
    if methods is None:
        methods = [m for m in ["purposive", "ig", "gradcam", "shap"] if m in summary]
    if metrics is None:
        metrics = list(METRIC_DISPLAY.keys())

    metrics = [m for m in metrics if any(m in summary[method] for method in methods)]

    n_metrics = len(metrics)
    n_methods = len(methods)
    bar_width = 0.18
    group_gap = 0.3

    fig, ax = plt.subplots(figsize=figsize)

    x_centers = np.arange(n_metrics) * (n_methods * bar_width + group_gap)

    for i, method in enumerate(methods):
        x_pos = x_centers + i * bar_width
        vals = []
        errs = []
        for metric in metrics:
            v = summary.get(method, {}).get(metric, float("nan"))
            e_key = metric.replace("_mean", "_std")
            e = summary.get(method, {}).get(e_key, 0.0)
            vals.append(v if not (isinstance(v, float) and np.isnan(v)) else 0.0)
            errs.append(e if not (isinstance(e, float) and np.isnan(e)) else 0.0)

        color = METHOD_COLORS.get(method, "gray")
        ax.bar(
            x_pos, vals, bar_width,
            yerr=errs, capsize=3,
            color=color, alpha=0.85,
            label=METHOD_LABELS.get(method, method),
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
        )

    ax.set_xticks(x_centers + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels([METRIC_DISPLAY.get(m, m) for m in metrics], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Image Explainability Methods Comparison (CUB-200-2011)", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 2: Qualitative visualisation grid
# ---------------------------------------------------------------------------

def plot_qualitative_grid(
    qual_dir: str,
    output_path: str,
    methods: Optional[List[str]] = None,
    n_images: int = 5,
    figsize_per_cell: Tuple[float, float] = (2.5, 2.5),
) -> None:
    """
    4 × n_images grid of qualitative saliency maps.

    Rows = methods, columns = curated images.
    """
    if methods is None:
        methods = ["purposive", "ig", "gradcam", "shap"]

    from PIL import Image

    # Discover available images for each method
    img_paths: Dict[str, List[str]] = {m: [] for m in methods}
    for m in methods:
        if not os.path.isdir(qual_dir):
            continue
        for fname in sorted(os.listdir(qual_dir)):
            if f"_{m}.png" in fname or f"_{m}.jpg" in fname:
                img_paths[m].append(os.path.join(qual_dir, fname))

    n_rows = len(methods)
    n_cols = n_images

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, method in enumerate(methods):
        for col in range(n_cols):
            ax = axes[row, col]
            paths = img_paths.get(method, [])
            if col < len(paths):
                try:
                    img = Image.open(paths[col])
                    ax.imshow(img)
                except Exception:  # noqa: BLE001
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

            ax.axis("off")
            if col == 0:
                ax.set_ylabel(METHOD_LABELS.get(method, method), fontsize=9, rotation=90, labelpad=4)

    plt.suptitle("Qualitative Saliency Map Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 3: Teleological flow diagram
# ---------------------------------------------------------------------------

def plot_teleological_flow(
    T_layers: List[np.ndarray],
    Delta_layers: List[np.ndarray],
    class_names: Optional[List[str]] = None,
    n_classes_to_show: int = 10,
    output_path: str = "teleological_flow.pdf",
    figsize: Tuple[float, float] = (14, 5),
) -> None:
    """
    Layer-wise discriminability flow diagram.

    Shows D_l as a heatmap over (class_i, class_j) pairs for each layer,
    and Delta_l as a bar chart below.

    Parameters
    ----------
    T_layers : list of np.ndarray
        Discriminability matrices, each (n_classes, n_classes).
    Delta_layers : list of np.ndarray
        Discriminability gain Delta_l = D_l - D_{l-1}.
    class_names : list of str or None
    n_classes_to_show : int
        Truncate the heatmap to the first n classes for readability.
    output_path : str
    """
    L = len(T_layers)
    if L == 0:
        logger.warning("No T_layers provided; skipping teleological flow figure.")
        return

    layer_labels = [f"L{l+1}" for l in range(L)]
    n_show = min(n_classes_to_show, T_layers[0].shape[0] if T_layers[0].ndim == 2 else 10)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, L, height_ratios=[3, 1], hspace=0.35, wspace=0.1)

    # Row 0: Discriminability heatmaps
    vmax = max(
        float(np.percentile(np.abs(T.flatten()), 95))
        for T in T_layers if T.size > 0
    )

    for l, T in enumerate(T_layers):
        ax = fig.add_subplot(gs[0, l])
        if T.ndim == 2:
            D_show = T[:n_show, :n_show]
        else:
            side = int(np.ceil(np.sqrt(len(T))))
            D_show = T[:side * side].reshape(side, side)[:n_show, :n_show]

        im = ax.imshow(D_show, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(layer_labels[l], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        if l == 0 and class_names:
            tick_labels = class_names[:n_show]
            ax.set_yticks(range(len(tick_labels)))
            ax.set_yticklabels(tick_labels, fontsize=5)

    # Colourbar on last heatmap
    plt.colorbar(im, ax=fig.axes[-1], fraction=0.046, pad=0.04, label="D_l")

    # Row 1: Delta bars
    for l, Delta in enumerate(Delta_layers):
        ax = fig.add_subplot(gs[1, l])
        mean_delta = float(np.mean(np.abs(Delta)))
        ax.bar([0], [mean_delta], color="#2196F3", alpha=0.8, width=0.6)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, None)
        ax.set_xticks([])
        ax.set_ylabel(r"$\Delta_l$" if l == 0 else "", fontsize=7)
        ax.tick_params(labelsize=7)

    plt.suptitle("Means-End Discriminability Flow", fontsize=12)
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 4: Sub-goal timeline plots
# ---------------------------------------------------------------------------

def plot_subgoal_timelines(
    trajectories: List[Dict[str, Any]],
    output_path: str,
    n_traj: int = 6,
    figsize: Tuple[float, float] = (14, 8),
    subgoal_colors: Optional[List[str]] = None,
) -> None:
    """
    Horizontal bar plots showing sub-goal assignments over time.

    Each row corresponds to one trajectory.  Segments are coloured by
    sub-goal identity.

    Parameters
    ----------
    trajectories : list of dict
        Each dict has 'actions' and 'subgoal_indices' (list/array of ints,
        one per timestep).
    output_path : str
    n_traj : int
        Maximum number of trajectories to plot.
    """
    if subgoal_colors is None:
        subgoal_colors = [
            "#F44336", "#2196F3", "#4CAF50", "#FF9800",
            "#9C27B0", "#00BCD4", "#795548", "#607D8B",
            "#E91E63", "#3F51B5",
        ]

    n_plot = min(n_traj, len(trajectories))
    fig, axes = plt.subplots(n_plot, 1, figsize=figsize, sharex=False)
    if n_plot == 1:
        axes = [axes]

    for idx in range(n_plot):
        traj = trajectories[idx]
        ax = axes[idx]

        actions = traj["actions"]
        sg_indices = traj.get("subgoal_indices", np.zeros(len(actions), dtype=int))
        T = len(actions)

        # Draw coloured segments
        prev_sg = int(sg_indices[0])
        seg_start = 0
        for t in range(1, T + 1):
            cur_sg = int(sg_indices[min(t, T - 1)])
            if cur_sg != prev_sg or t == T:
                color = subgoal_colors[prev_sg % len(subgoal_colors)]
                ax.barh(
                    0, t - seg_start,
                    left=seg_start, height=0.6,
                    color=color, alpha=0.8,
                )
                seg_start = t
                prev_sg = cur_sg

        ax.set_xlim(0, T)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_ylabel(f"Traj {idx + 1}", fontsize=8, rotation=0, ha="right", va="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx < n_plot - 1:
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])

    axes[-1].set_xlabel("Timestep", fontsize=10)
    plt.suptitle("Sub-Goal Segment Timeline", fontsize=12)
    plt.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fmt = Path(output_path).suffix.lstrip(".") or "pdf"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", format=fmt)
    plt.close(fig)
    logger.info("Saved figure to %s", output_path)


def _load_image_summary(results_dir: str) -> Dict[str, Dict[str, float]]:
    """Load image_summary.json or fall back to image_summary.csv."""
    json_path = os.path.join(results_dir, "image_summary.json")
    csv_path  = os.path.join(results_dir, "image_summary.csv")

    if os.path.exists(json_path):
        with open(json_path) as f:
            raw = json.load(f)
        # Replace None with nan
        out: Dict[str, Dict[str, float]] = {}
        for method, metrics in raw.items():
            out[method] = {
                k: float(v) if v is not None else float("nan")
                for k, v in metrics.items()
            }
        return out

    if os.path.exists(csv_path):
        summary: Dict[str, Dict[str, float]] = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = row.pop("method")
                summary[method] = {k: float(v) for k, v in row.items() if v not in ("", "nan")}
        return summary

    logger.warning("No image summary found in %s", results_dir)
    return {}


def _load_rl_results(results_dir: str) -> Dict[str, Any]:
    path = os.path.join(results_dir, "rl_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    logger.warning("No rl_results.json found in %s", results_dir)
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-quality figures for teleological XAI."
    )
    parser.add_argument("--image-results", default="results/image/")
    parser.add_argument("--rl-results",    default="results/rl/")
    parser.add_argument("--output-dir",    default="results/figures/")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # ---- Figure 1: Comparison bar chart -----------------------------------
    image_summary = _load_image_summary(args.image_results)
    if image_summary:
        plot_comparison_bar(
            image_summary,
            output_path=os.path.join(out, "comparison_bar.pdf"),
        )
    else:
        logger.warning("Skipping comparison bar chart (no image summary).")

    # ---- Figure 2: Qualitative grid ---------------------------------------
    qual_dir = os.path.join(args.image_results, "qualitative")
    if os.path.isdir(qual_dir):
        plot_qualitative_grid(
            qual_dir=qual_dir,
            output_path=os.path.join(out, "qualitative_grid.pdf"),
        )
    else:
        logger.warning("No qualitative dir at %s; skipping.", qual_dir)

    # ---- Figure 3: Teleological flow (stub data) --------------------------
    # Try to load pre-computed T/Delta arrays; generate synthetic if absent.
    flow_path = os.path.join(args.image_results, "means_end_layers.npz")
    if os.path.exists(flow_path):
        data = np.load(flow_path, allow_pickle=True)
        T_layers     = list(data["T_layers"])
        Delta_layers = list(data["Delta_layers"])
    else:
        logger.warning("means_end_layers.npz not found; using synthetic data for flow plot.")
        n_cls = 10
        T_layers = [np.random.rand(n_cls, n_cls) * (l + 1) / 5 for l in range(5)]
        Delta_layers = [T_layers[0]] + [T_layers[l] - T_layers[l-1] for l in range(1, 5)]

    plot_teleological_flow(
        T_layers=T_layers,
        Delta_layers=Delta_layers,
        output_path=os.path.join(out, "teleological_flow.pdf"),
    )

    # ---- Figure 4: Sub-goal timelines ------------------------------------
    rl_env_dirs = [
        d for d in (
            os.path.join(args.rl_results, e)
            for e in os.listdir(args.rl_results)
            if os.path.isdir(os.path.join(args.rl_results, e))
        )
    ] if os.path.isdir(args.rl_results) else []

    traj_for_plot: List[Dict[str, Any]] = []
    for env_dir in rl_env_dirs:
        traj_path = os.path.join(env_dir, "trajectories.npz")
        if os.path.exists(traj_path):
            raw = np.load(traj_path, allow_pickle=True)
            for item in raw["trajectories"]:
                traj_for_plot.append(item)
            if len(traj_for_plot) >= 6:
                break

    if not traj_for_plot:
        # Synthetic
        logger.warning("No RL trajectories found; generating synthetic timeline data.")
        for _ in range(6):
            T = np.random.randint(50, 120)
            sg = np.repeat(np.arange(np.random.randint(3, 8)), T // np.random.randint(3, 8) + 1)[:T]
            traj_for_plot.append({"actions": np.zeros(T, int), "subgoal_indices": sg})

    plot_subgoal_timelines(
        trajectories=traj_for_plot,
        output_path=os.path.join(out, "subgoal_timeline.pdf"),
    )

    logger.info("All figures written to %s", out)


if __name__ == "__main__":
    main()
