"""
Confusion set construction for teleological XAI on CUB-200-2011.

For input x with true class c, the confusion set is:

    C(x) = {j != c : p(j|x) >= tau}

Fall back to top-k_min if |C(x)| < k_min, truncate to k_max if too large.
Default hyperparameters: tau=0.02, k_min=3, k_max=10.

Confusion weight for j in C(x):

    w_j(x) = alpha  * (p(j|x)       / sum_{j'} p(j'|x))
            + (1-alpha) * ([p_c]_j / sum_{j'} [p_c]_{j'})

where alpha=0.7 blends instance-level and global confusion distributions.

Global confusion matrix M (200x200):
    M[i,j] = # times class-i image got highest non-i logit for class j
Row-normalise M to obtain per-class confusion distributions p_c.

Public API
----------
compute_global_confusion_matrix(model, train_loader, device) -> np.ndarray
compute_instance_confusion_set(logits, true_class, global_dist,
                               tau, k_min, k_max, alpha) -> (set, dict)
precompute_and_cache(model, loader, device, cache_path)
load_confusion_cache(cache_path) -> dict
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Default hyperparameters (matching image_config.yaml)
# ---------------------------------------------------------------------------
DEFAULT_TAU: float = 0.02
DEFAULT_K_MIN: int = 3
DEFAULT_K_MAX: int = 10
DEFAULT_ALPHA: float = 0.7
NUM_CLASSES: int = 200


# ---------------------------------------------------------------------------
# Global confusion matrix
# ---------------------------------------------------------------------------

def compute_global_confusion_matrix(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Run inference on *train_loader* and compute the 200x200 global confusion
    matrix M, then row-normalise it.

    M[i, j] = number of times a class-i image received its highest non-i
              predicted logit for class j.

    The returned array is row-normalised (each row sums to 1.0), yielding a
    per-class confusion distribution p_c over competitor classes.

    Parameters
    ----------
    model : nn.Module
        Fine-tuned ResNet-50. Must output raw logits of shape (B, 200).
        Will be put into eval mode internally.
    train_loader : DataLoader
        Yields (images, labels) batches.  Labels are 0-indexed integers.
    device : torch.device

    Returns
    -------
    M : np.ndarray, shape (200, 200), dtype float32
        Row-normalised confusion matrix.  M[i, i] = 0 by construction.
    """
    model.eval()
    model.to(device)

    # Raw count matrix
    M_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Building confusion matrix"):
            images = images.to(device)
            labels = labels.to(device)  # (B,)

            logits = model(images)  # (B, 200)

            for b in range(logits.size(0)):
                true_c = labels[b].item()
                row_logits = logits[b].clone()
                # Mask out the true class to find top non-c competitor
                row_logits[true_c] = float("-inf")
                top_competitor = row_logits.argmax().item()
                M_counts[true_c, top_competitor] += 1

    # Row-normalise; rows with zero counts stay zero
    row_sums = M_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid divide-by-zero
    M_normed = (M_counts / row_sums).astype(np.float32)

    return M_normed


# ---------------------------------------------------------------------------
# Instance-level confusion set
# ---------------------------------------------------------------------------

def compute_instance_confusion_set(
    logits: torch.Tensor,
    true_class: int,
    global_dist: np.ndarray,
    tau: float = DEFAULT_TAU,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[Set[int], Dict[int, float]]:
    """
    Compute the confusion set and per-class weights for a single input.

    Parameters
    ----------
    logits : torch.Tensor, shape (200,)
        Raw model logits for a single image (before softmax).
    true_class : int
        0-indexed ground-truth class label.
    global_dist : np.ndarray, shape (200,)
        Row true_class of the row-normalised global confusion matrix
        (M[true_class, :]).
    tau : float
        Threshold on softmax probability for inclusion in the confusion set.
    k_min : int
        Minimum confusion set size (fallback to top-k_min).
    k_max : int
        Maximum confusion set size (truncated to top-k_max).
    alpha : float
        Blending weight: 1.0 = purely instance-level, 0.0 = purely global.

    Returns
    -------
    confusion_set : set of int
        Set of competitor class indices.
    weights : dict mapping int -> float
        Normalised blended weight w_j(x) for each j in confusion_set.
        Values sum to 1.0.
    """
    # Compute softmax probabilities
    probs = F.softmax(logits.float(), dim=0).cpu().numpy()  # (200,)

    # Build candidate list (all classes except true_class)
    candidate_mask = np.ones(NUM_CLASSES, dtype=bool)
    candidate_mask[true_class] = False
    candidate_indices = np.where(candidate_mask)[0]

    # --- Step 1: threshold-based confusion set ---
    above_tau = candidate_indices[probs[candidate_indices] >= tau]

    if len(above_tau) < k_min:
        # Fall back: take top-k_min by probability
        sorted_by_prob = candidate_indices[
            np.argsort(probs[candidate_indices])[::-1]
        ]
        confusion_set_arr = sorted_by_prob[:k_min]
    elif len(above_tau) > k_max:
        # Truncate: keep top-k_max by probability among those above tau
        sorted_above = above_tau[np.argsort(probs[above_tau])[::-1]]
        confusion_set_arr = sorted_above[:k_max]
    else:
        confusion_set_arr = above_tau

    confusion_set: Set[int] = set(int(j) for j in confusion_set_arr)

    # --- Step 2: compute blended weights ---
    instance_probs = np.array([probs[j] for j in confusion_set_arr], dtype=np.float64)
    global_probs   = np.array([global_dist[j] for j in confusion_set_arr], dtype=np.float64)

    # Normalise each component over the confusion set
    instance_sum = instance_probs.sum()
    global_sum   = global_probs.sum()

    instance_norm = (
        instance_probs / instance_sum if instance_sum > 0 else np.ones_like(instance_probs) / len(instance_probs)
    )
    global_norm = (
        global_probs / global_sum if global_sum > 0 else np.ones_like(global_probs) / len(global_probs)
    )

    blended = alpha * instance_norm + (1.0 - alpha) * global_norm

    # Final normalisation (blended already sums to ~1 if both components do,
    # but re-normalise for numerical safety)
    blended_sum = blended.sum()
    if blended_sum > 0:
        blended = blended / blended_sum

    weights: Dict[int, float] = {
        int(j): float(w) for j, w in zip(confusion_set_arr, blended)
    }

    return confusion_set, weights


# ---------------------------------------------------------------------------
# Precompute and cache
# ---------------------------------------------------------------------------

def precompute_and_cache(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cache_path: str,
    tau: float = DEFAULT_TAU,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    alpha: float = DEFAULT_ALPHA,
    global_confusion_matrix: Optional[np.ndarray] = None,
) -> None:
    """
    Precompute global confusion matrix and per-image confusion sets, then
    save everything to a single .npz cache file.

    The function assumes *loader* yields ``(images, labels, img_ids)`` tuples.
    If the DataLoader yields only ``(images, labels)``, integer indices
    ``0, 1, 2, ...`` are used as image IDs.

    Parameters
    ----------
    model : nn.Module
        Fine-tuned ResNet-50 (eval mode).
    loader : DataLoader
        Typically the *training* loader for the global confusion matrix and
        the *test* loader for per-image confusion sets, but this function
        computes everything from *loader* in a single pass.

        For a two-pass workflow (train for M, test for C(x)), call this
        function twice or use the ``global_confusion_matrix`` argument to
        supply a pre-built M from the training set.
    device : torch.device
    cache_path : str
        Destination .npz file path.  Parent directories are created if needed.
    tau, k_min, k_max, alpha : float / int
        Hyperparameters forwarded to :func:`compute_instance_confusion_set`.
    global_confusion_matrix : np.ndarray or None
        If provided, skip building the global confusion matrix and use this
        pre-computed one instead.  Shape must be (200, 200).

    Saved keys in the .npz
    ----------------------
    ``global_confusion_matrix`` : (200, 200) float32
    ``img_ids``                 : (N,) int64
    ``confusion_sets``          : stored as a pickled bytes object under key
                                  ``confusion_sets_pkl``
    ``weights``                 : stored as a pickled bytes object under key
                                  ``weights_pkl``
    ``hyperparams``             : pickled dict
    """
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)

    model.eval()
    model.to(device)

    # ------------------------------------------------------------------
    # Pass 1: collect logits + labels for every image in loader
    # ------------------------------------------------------------------
    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []
    all_img_ids: list[int] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(loader, desc="Collecting logits")
        ):
            if len(batch) == 3:
                images, labels, img_ids = batch
                img_ids_list = img_ids.tolist()
            else:
                images, labels = batch
                # Assign sequential IDs based on batch position
                start = batch_idx * loader.batch_size  # type: ignore[arg-type]
                img_ids_list = list(range(start, start + len(labels)))

            images = images.to(device)
            logits = model(images)  # (B, 200)

            all_logits.append(logits.cpu().numpy())
            all_labels.extend(labels.tolist())
            all_img_ids.extend(img_ids_list)

    all_logits_arr = np.concatenate(all_logits, axis=0)   # (N, 200)
    all_labels_arr = np.array(all_labels, dtype=np.int64)  # (N,)
    all_img_ids_arr = np.array(all_img_ids, dtype=np.int64)  # (N,)

    # ------------------------------------------------------------------
    # Build or reuse global confusion matrix
    # ------------------------------------------------------------------
    if global_confusion_matrix is None:
        print("Computing global confusion matrix from loader ...")
        M = _build_confusion_matrix_from_arrays(all_logits_arr, all_labels_arr)
    else:
        M = global_confusion_matrix.astype(np.float32)

    # ------------------------------------------------------------------
    # Pass 2: compute per-image confusion sets from cached logits
    # ------------------------------------------------------------------
    confusion_sets_dict: Dict[int, list] = {}
    weights_dict_outer: Dict[int, Dict[int, float]] = {}

    for i in tqdm(range(len(all_labels_arr)), desc="Computing confusion sets"):
        img_id = int(all_img_ids_arr[i])
        true_c = int(all_labels_arr[i])
        logits_tensor = torch.from_numpy(all_logits_arr[i])
        global_dist = M[true_c]

        c_set, w_dict = compute_instance_confusion_set(
            logits_tensor, true_c, global_dist, tau, k_min, k_max, alpha
        )
        confusion_sets_dict[img_id] = sorted(c_set)
        weights_dict_outer[img_id] = w_dict

    # ------------------------------------------------------------------
    # Serialize and save
    # ------------------------------------------------------------------
    hyperparams = dict(tau=tau, k_min=k_min, k_max=k_max, alpha=alpha)

    # Pickle variable-length structures
    confusion_sets_bytes = pickle.dumps(confusion_sets_dict)
    weights_bytes = pickle.dumps(weights_dict_outer)
    hyperparams_bytes = pickle.dumps(hyperparams)

    np.savez_compressed(
        cache_path,
        global_confusion_matrix=M,
        img_ids=all_img_ids_arr,
        confusion_sets_pkl=np.frombuffer(confusion_sets_bytes, dtype=np.uint8),
        weights_pkl=np.frombuffer(weights_bytes, dtype=np.uint8),
        hyperparams_pkl=np.frombuffer(hyperparams_bytes, dtype=np.uint8),
    )

    print(f"Saved confusion cache to {cache_path}  ({len(all_labels_arr)} images)")


# ---------------------------------------------------------------------------
# Load cache
# ---------------------------------------------------------------------------

def load_confusion_cache(cache_path: str) -> dict:
    """
    Load a confusion cache file produced by :func:`precompute_and_cache`.

    Parameters
    ----------
    cache_path : str
        Path to the .npz file.

    Returns
    -------
    dict with keys:
        ``"global_confusion_matrix"`` : np.ndarray, shape (200, 200)
        ``"img_ids"``                 : np.ndarray, shape (N,)
        ``"confusion_sets"``          : dict {img_id -> list of int}
        ``"weights"``                 : dict {img_id -> {competitor -> float}}
        ``"hyperparams"``             : dict {tau, k_min, k_max, alpha}
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Confusion cache not found: {cache_path}")

    data = np.load(cache_path, allow_pickle=False)

    M = data["global_confusion_matrix"]
    img_ids = data["img_ids"]
    confusion_sets = pickle.loads(data["confusion_sets_pkl"].tobytes())
    weights = pickle.loads(data["weights_pkl"].tobytes())
    hyperparams = pickle.loads(data["hyperparams_pkl"].tobytes())

    return {
        "global_confusion_matrix": M,
        "img_ids": img_ids,
        "confusion_sets": confusion_sets,
        "weights": weights,
        "hyperparams": hyperparams,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_confusion_matrix_from_arrays(
    logits_arr: np.ndarray,   # (N, 200)
    labels_arr: np.ndarray,   # (N,)
) -> np.ndarray:
    """Build and row-normalise M without needing a DataLoader / model pass."""
    M_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    for i in range(len(labels_arr)):
        true_c = int(labels_arr[i])
        row = logits_arr[i].copy()
        row[true_c] = -np.inf
        top_competitor = int(np.argmax(row))
        M_counts[true_c, top_competitor] += 1

    row_sums = M_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return (M_counts / row_sums).astype(np.float32)


def get_confusion_set_for_image(
    cache: dict,
    img_id: int,
) -> Tuple[list, Dict[int, float]]:
    """
    Retrieve the pre-computed confusion set and weights for a single image.

    Parameters
    ----------
    cache : dict
        Output of :func:`load_confusion_cache`.
    img_id : int
        Image ID key.

    Returns
    -------
    (confusion_set_list, weights_dict)
    """
    confusion_set = cache["confusion_sets"].get(img_id, [])
    weights = cache["weights"].get(img_id, {})
    return confusion_set, weights
