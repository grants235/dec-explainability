"""
Image evaluation pipeline for teleological explainability on CUB-200-2011.

Runs all image metrics (PBPA, Deletion AUC, Insertion AUC, Purposive
Specificity, Means-End Coherence) for each saliency method and writes:

  - ``results_dir/image_results.csv``    – per-image raw scores
  - ``results_dir/image_summary.csv``    – aggregated mean ± std per method
  - ``results_dir/image_summary.tex``    – LaTeX table
  - ``results_dir/qualitative/``         – visualisations for curated images
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import (
    compute_deletion_auc,
    compute_insertion_auc,
    compute_pbpa,
    compute_purposive_specificity,
    compute_means_end_coherence,
    compute_diagnostic_parts,
    CUB_PART_NAMES,
    PART_TO_ATTR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Indices of images (within the test set) to include in qualitative output
CURATED_IMAGE_INDICES: List[int] = list(range(20))

# Methods to evaluate (must match keys in ``saliency_methods`` dict built
# inside ``run_image_evaluation``)
METHOD_NAMES: List[str] = ["purposive", "ig", "gradcam", "shap"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_part_to_attr_indices(
    attribute_annotations: np.ndarray,
    attr_part_labels: Optional[List[str]] = None,
) -> Dict[str, List[int]]:
    """
    Build a mapping from CUB part name -> list of attribute column indices.

    If ``attr_part_labels`` is provided it must be a list of length
    ``n_attributes`` where each entry is the canonical part name that
    attribute corresponds to (using the values of PART_TO_ATTR).

    Otherwise we return an empty dict (PBPA cannot be computed).
    """
    if attr_part_labels is None:
        return {}

    part_to_idxs: Dict[str, List[int]] = {p: [] for p in PART_TO_ATTR.values()}
    for idx, label in enumerate(attr_part_labels):
        if label in part_to_idxs:
            part_to_idxs[label].append(idx)
    return part_to_idxs


def _build_dataset_mean(test_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Compute per-channel dataset mean over the test set as a (1,3,1,1) tensor."""
    channel_sum = torch.zeros(3)
    n_images = 0
    for batch in test_loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        channel_sum += imgs.mean(dim=[0, 2, 3]).cpu() * imgs.shape[0]
        n_images += imgs.shape[0]
    mean = (channel_sum / n_images).view(1, 3, 1, 1).to(device)
    return mean


def _get_saliency_map(
    method_name: str,
    x: torch.Tensor,
    target_class: int,
    purposive_saliency: Any,
    ig_explainer: Any,
    gradcam_explainer: Any,
    shap_explainer: Any,
    confusion_cache: Dict[int, List[int]],
    class_idx: int,
    means_end_decomp: Optional[Any] = None,
) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    """
    Dispatch to the appropriate explainer and return:
        (saliency_map, per_competitor_maps)

    ``per_competitor_maps`` is only populated for the purposive method.
    """
    if method_name == "purposive":
        competitors = confusion_cache.get(class_idx, [])
        if not competitors:
            return np.zeros((448, 448), dtype=np.float32), []

        # Equal weights for each competitor
        weights = {j: 1.0 / len(competitors) for j in competitors}

        # PurposiveSaliency.compute() returns (per_competitor_maps, s_agg, annotation_map)
        per_comp_maps_raw, s_agg_raw, _ = purposive_saliency.compute(
            x, class_idx, competitors, weights
        )

        # s_agg: Tensor (H, W) — the aggregated map
        if isinstance(s_agg_raw, torch.Tensor):
            agg_map = s_agg_raw.detach().cpu().float().numpy()
        else:
            agg_map = np.asarray(s_agg_raw, dtype=np.float32)

        # per_competitor_maps: dict {class_idx: Tensor (H, W)}
        per_comp_maps: List[np.ndarray] = []
        for j in competitors:
            m = per_comp_maps_raw.get(j) if isinstance(per_comp_maps_raw, dict) else None
            if m is None:
                continue
            if isinstance(m, torch.Tensor):
                m = m.detach().cpu().float().numpy()
            m = np.asarray(m, dtype=np.float32)
            if m.ndim != 2:
                m = m.reshape(448, 448) if m.size == 448 * 448 else m.squeeze()
            per_comp_maps.append(m)

        return agg_map, per_comp_maps

    elif method_name == "ig":
        m = ig_explainer.compute(x, target_class)
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        return m.astype(np.float32), None

    elif method_name == "gradcam":
        m = gradcam_explainer.compute(x, target_class)
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        return m.astype(np.float32), None

    elif method_name == "shap":
        m = shap_explainer.compute(x, target_class)
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        return m.astype(np.float32), None

    else:
        raise ValueError(f"Unknown method: {method_name}")


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_image_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    confusion_cache: Dict[int, List[int]],
    purposive_saliency: Any,
    ig_explainer: Any,
    gradcam_explainer: Any,
    shap_explainer: Any,
    means_end_decomp: Optional[Any],
    part_annotations: Dict[int, Dict[str, Optional[Tuple[int, int]]]],
    attribute_annotations: np.ndarray,
    class_attribute_matrix: np.ndarray,
    part_names: List[str],
    class_names: List[str],
    config: Dict[str, Any],
    device: torch.device,
    results_dir: str,
    attr_part_labels: Optional[List[str]] = None,
    curated_indices: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run all image evaluation metrics and generate result tables / figures.

    Parameters
    ----------
    model : nn.Module
        Trained ResNet-50 classifier (eval mode).
    test_loader : DataLoader
        Yields ``(image_tensor, label)`` batches.  Shuffle=False.
    confusion_cache : dict
        Mapping class_idx -> list of competitor class indices.
    purposive_saliency : object
        Has a ``.compute(x, class_i, class_j)`` method.
    ig_explainer : IGExplainer
    gradcam_explainer : GradCAMExplainer
    shap_explainer : SHAPExplainer
    means_end_decomp : object or None
        If provided must have:
        - ``.T_layers``     : list of np.ndarray  (L layers)
        - ``.Delta_layers`` : list of np.ndarray
        - ``.kappa_layers`` : list of np.ndarray
    part_annotations : dict
        image_idx -> dict mapping part_name -> (row, col) or None.
    attribute_annotations : np.ndarray
        Shape ``(n_images_test, n_attributes)`` binary.
    class_attribute_matrix : np.ndarray
        Shape ``(n_classes, n_attributes)``.
    part_names : list of str
        CUB canonical part names (length 15).
    class_names : list of str
        Human-readable class labels.
    config : dict
        Merged YAML config as a plain dict.
    device : torch.device
    results_dir : str
        Root directory for saving outputs.
    attr_part_labels : list of str or None
        Per-attribute part name (length n_attributes).
    curated_indices : list of int or None
        Indices into the test set for qualitative visualisation.

    Returns
    -------
    dict
        Nested dict: method_name -> metric_name -> mean_score.
    """
    model.eval()
    model.to(device)

    if curated_indices is None:
        curated_indices = CURATED_IMAGE_INDICES

    os.makedirs(results_dir, exist_ok=True)
    qual_dir = os.path.join(results_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    # Evaluation config
    eval_cfg = config.get("evaluation", {})
    deletion_steps = int(eval_cfg.get("deletion_steps", 21))
    insertion_steps = int(eval_cfg.get("insertion_steps", 21))
    pbpa_percentile = float(eval_cfg.get("pbpa_saliency_threshold_percentile", 90))

    part_to_attr_indices = _make_part_to_attr_indices(attribute_annotations, attr_part_labels)

    # Dataset mean for deletion masking
    dataset_mean = _build_dataset_mean(test_loader, device)

    # Storage: method -> list of per-image metric dicts
    per_image_results: Dict[str, List[Dict[str, float]]] = {m: [] for m in METHOD_NAMES}

    # Means-End Coherence is global (not per-image); compute once if available
    mec_results: Dict[str, float] = {}
    if means_end_decomp is not None:
        try:
            mec_results = compute_means_end_coherence(
                means_end_decomp.T_layers,
                means_end_decomp.Delta_layers,
                means_end_decomp.kappa_layers,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Means-End Coherence computation failed: %s", exc)

    # -----------------------------------------------------------------------
    # Main loop over test images
    # -----------------------------------------------------------------------
    global_image_idx = 0
    for batch_imgs, batch_labels in tqdm(test_loader, desc="Evaluating"):
        batch_imgs = batch_imgs.to(device)

        for b in range(batch_imgs.shape[0]):
            x = batch_imgs[b : b + 1]  # (1, 3, H, W)
            true_class = int(batch_labels[b].item())
            image_idx = global_image_idx
            global_image_idx += 1

            # Model prediction
            with torch.no_grad():
                logits = model(x)
                pred_class = int(logits.argmax(dim=-1).item())

            # Skip misclassified images for metric computation
            # (saliency quality is best evaluated on correct predictions)
            if pred_class != true_class:
                continue

            # Part keypoints for this image
            kpts = part_annotations.get(image_idx, {})

            # Diagnostic parts for the confusion set
            competitors = confusion_cache.get(true_class, [])
            all_diagnostic_parts: List[str] = []
            for comp in competitors:
                dp, _ = compute_diagnostic_parts(
                    true_class, comp, class_attribute_matrix, part_to_attr_indices
                )
                all_diagnostic_parts.extend(dp)
            diagnostic_parts = list(set(all_diagnostic_parts))

            for method in METHOD_NAMES:
                try:
                    sal_map, per_comp_maps = _get_saliency_map(
                        method, x, true_class,
                        purposive_saliency, ig_explainer, gradcam_explainer, shap_explainer,
                        confusion_cache, true_class, means_end_decomp,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Saliency computation failed for method=%s img=%d: %s",
                                   method, image_idx, exc)
                    continue

                # PBPA
                pbpa = compute_pbpa(
                    sal_map, kpts, diagnostic_parts,
                    image_size=448, radius=30, percentile=pbpa_percentile,
                ) if diagnostic_parts else float("nan")

                # Deletion / Insertion AUC
                del_auc = compute_deletion_auc(
                    model, x, sal_map, true_class, device,
                    n_steps=deletion_steps, mask_value=dataset_mean,
                )
                ins_auc = compute_insertion_auc(
                    model, x, sal_map, true_class, device,
                    n_steps=insertion_steps,
                )

                # Purposive Specificity (only for purposive method)
                ps = float("nan")
                if method == "purposive" and per_comp_maps and len(per_comp_maps) >= 2:
                    ps = compute_purposive_specificity(per_comp_maps)

                row = {
                    "image_idx":   image_idx,
                    "true_class":  true_class,
                    "pbpa":        pbpa,
                    "deletion_auc": del_auc,
                    "insertion_auc": ins_auc,
                    "purposive_specificity": ps,
                }
                per_image_results[method].append(row)

                # Qualitative visualisation for curated images
                if image_idx in curated_indices:
                    _save_qualitative(
                        method, x, sal_map,
                        class_names[true_class] if true_class < len(class_names) else str(true_class),
                        qual_dir, image_idx,
                        ig_explainer, gradcam_explainer, shap_explainer,
                    )

    # -----------------------------------------------------------------------
    # Aggregate and save
    # -----------------------------------------------------------------------
    summary = _aggregate_results(per_image_results, mec_results)
    _save_csv(per_image_results, results_dir)
    _save_summary_csv(summary, results_dir)
    _save_latex_table(summary, results_dir)

    logger.info("Image evaluation complete. Results in %s", results_dir)
    return summary


# ---------------------------------------------------------------------------
# Qualitative visualisation dispatcher
# ---------------------------------------------------------------------------

def _save_qualitative(
    method: str,
    x: torch.Tensor,
    sal_map: np.ndarray,
    class_name: str,
    qual_dir: str,
    image_idx: int,
    ig_explainer: Any,
    gradcam_explainer: Any,
    shap_explainer: Any,
) -> None:
    """Save qualitative visualisation for a single (method, image) pair."""
    save_path = os.path.join(qual_dir, f"img{image_idx:04d}_{method}.png")
    try:
        if method == "purposive":
            _visualize_generic(x, sal_map, class_name, save_path)
        elif method == "ig" and ig_explainer is not None:
            ig_explainer.visualize(x, sal_map, class_name, save_path)
        elif method == "gradcam" and gradcam_explainer is not None:
            gradcam_explainer.visualize(x, sal_map, class_name, save_path)
        elif method == "shap" and shap_explainer is not None:
            shap_explainer.visualize(x, sal_map, class_name, save_path)
        else:
            _visualize_generic(x, sal_map, class_name, save_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qualitative viz failed for method=%s img=%d: %s", method, image_idx, exc)


def _visualize_generic(
    x: torch.Tensor,
    sal_map: np.ndarray,
    class_name: str,
    save_path: str,
) -> None:
    """Simple overlay visualisation without a specific explainer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if x.dim() == 4:
        x = x.squeeze(0)

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_np = (x.detach().cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()

    cmap = cm.get_cmap("hot")
    hm_colored = cmap(sal_map)[..., :3]
    overlay = np.clip(0.6 * img_np + 0.4 * hm_colored, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np); axes[0].axis("off"); axes[0].set_title("Original")
    axes[1].imshow(sal_map, cmap="hot"); axes[1].axis("off"); axes[1].set_title("Saliency")
    axes[2].imshow(overlay); axes[2].axis("off"); axes[2].set_title(class_name)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_results(
    per_image_results: Dict[str, List[Dict[str, float]]],
    mec_results: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Compute mean ± std per method per metric."""
    metric_keys = ["pbpa", "deletion_auc", "insertion_auc", "purposive_specificity"]
    summary: Dict[str, Dict[str, float]] = {}

    for method, rows in per_image_results.items():
        method_summary: Dict[str, float] = {}
        for mk in metric_keys:
            vals = [r[mk] for r in rows if not np.isnan(r.get(mk, float("nan")))]
            method_summary[f"{mk}_mean"] = float(np.mean(vals)) if vals else float("nan")
            method_summary[f"{mk}_std"] = float(np.std(vals)) if vals else float("nan")

        # Attach Means-End Coherence (global, same for all methods)
        for k, v in mec_results.items():
            method_summary[k] = v

        summary[method] = method_summary

    return summary


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _save_csv(
    per_image_results: Dict[str, List[Dict[str, float]]],
    results_dir: str,
) -> None:
    """Write flat per-image results CSV."""
    path = os.path.join(results_dir, "image_results.csv")
    fieldnames = [
        "method", "image_idx", "true_class",
        "pbpa", "deletion_auc", "insertion_auc", "purposive_specificity",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in per_image_results.items():
            for row in rows:
                writer.writerow({"method": method, **row})
    logger.info("Saved per-image CSV to %s", path)


def _save_summary_csv(
    summary: Dict[str, Dict[str, float]],
    results_dir: str,
) -> None:
    """Write aggregated summary CSV."""
    path = os.path.join(results_dir, "image_summary.csv")
    if not summary:
        return
    all_keys = sorted({k for d in summary.values() for k in d})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + all_keys)
        writer.writeheader()
        for method, metrics in summary.items():
            writer.writerow({"method": method, **metrics})
    logger.info("Saved summary CSV to %s", path)


def _save_latex_table(
    summary: Dict[str, Dict[str, float]],
    results_dir: str,
) -> None:
    """Write a LaTeX booktabs table from the summary dict."""
    path = os.path.join(results_dir, "image_summary.tex")

    # Choose which metric/columns to show
    col_defs = [
        ("PBPA",    "pbpa_mean",              "pbpa_std"),
        ("Del AUC", "deletion_auc_mean",      "deletion_auc_std"),
        ("Ins AUC", "insertion_auc_mean",     "insertion_auc_std"),
        ("PS",      "purposive_specificity_mean", "purposive_specificity_std"),
        ("MEC",     "coherence",              None),
    ]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Image explainability evaluation on CUB-200-2011.}",
        r"\label{tab:image_results}",
        r"\begin{tabular}{l" + "c" * len(col_defs) + "}",
        r"\toprule",
        "Method & " + " & ".join(c[0] for c in col_defs) + r" \\",
        r"\midrule",
    ]

    method_display = {
        "purposive": "Purposive (ours)",
        "ig":        "Integrated Gradients",
        "gradcam":   "GradCAM",
        "shap":      "GradientSHAP",
    }

    for method in METHOD_NAMES:
        if method not in summary:
            continue
        m = summary[method]
        cells = [method_display.get(method, method)]
        for _, mean_key, std_key in col_defs:
            mv = m.get(mean_key, float("nan"))
            sv = m.get(std_key, float("nan")) if std_key else None
            if np.isnan(mv):
                cells.append("--")
            elif sv is not None and not np.isnan(sv):
                cells.append(f"{mv:.3f}$\\pm${sv:.3f}")
            else:
                cells.append(f"{mv:.3f}")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved LaTeX table to %s", path)
