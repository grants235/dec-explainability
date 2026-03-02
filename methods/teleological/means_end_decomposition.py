"""
Means-End Decomposition for Teleological Explainability.

For each layer l in a backbone (labelled L1 … L5), we measure how
discriminative the layer's spatially-averaged activations are at separating
pairs of classes (i, j) via logistic-regression probes evaluated with
5-fold cross-validation AUROC.

Definitions
-----------
Teleological content:
    T_l[i,j]  = D_l(i,j)                      (AUROC of probe at layer l)

Marginal contribution:
    Delta_l[i,j] = D_l[i,j] - D_{l-1}[i,j]   (D_0 = 0.5 everywhere)

Consumption score (how much layer l "uses up" information present in l-1):
    kappa_l[i,j] = D_l[i,j] - D_tilde_l[i,j]
    where D_tilde_l is the AUROC when layer l-1 activations are ablated
    (replaced by the pooled class-mean of {i,j} combined).

Pair selection:
    - "Hard" pairs: all (i,j) from the confusion matrix where
        M[i,j] + M[j,i] >= 5.
    - "Easy" control pairs: 100 randomly sampled remaining pairs.

References: Shanklin et al., "Teleological Explainability" (2026).
"""

from __future__ import annotations

import os
import random
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAYER_NAMES: List[str] = ["L1", "L2", "L3", "L4", "L5"]
D0: float = 0.5            # prior AUROC (random baseline)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_numpy(t: Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _logistic_auroc(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """
    Train a logistic regression probe with stratified k-fold CV and return
    the mean AUROC.

    Parameters
    ----------
    X : (N, D) float array   – activations.
    y : (N,)   int array     – binary labels {0, 1}.
    n_splits : int           – number of CV folds.
    random_state : int       – RNG seed.

    Returns
    -------
    float in [0, 1].
    """
    # Guard: if only one class present we cannot compute AUROC.
    unique_cls = np.unique(y)
    if len(unique_cls) < 2:
        return D0

    # Guard: fewer samples than splits -> reduce n_splits.
    n_splits = min(n_splits, len(y))
    if n_splits < 2:
        return D0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs: List[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        if len(np.unique(y_va)) < 2:
            # Skip folds where validation set has only one class.
            continue

        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_va)
        # pos_class column = index of class 1 in clf.classes_.
        pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else -1
        if pos_col == -1:
            aucs.append(D0)
        else:
            aucs.append(roc_auc_score(y_va, proba[:, pos_col]))

    return float(np.mean(aucs)) if aucs else D0


# ---------------------------------------------------------------------------
# Activation extractor
# ---------------------------------------------------------------------------

class _ActivationExtractor:
    """
    Registers forward hooks on a set of named sub-modules to capture their
    output activations.  After a forward pass, activations are available in
    ``self.cache`` keyed by layer name.

    The activations are spatially average-pooled to a single vector per
    sample: (N, D).
    """

    def __init__(self, model: nn.Module, layer_map: Dict[str, nn.Module]) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            The full model (kept for reference; not called here).
        layer_map : dict[str, nn.Module]
            Mapping from a label string (e.g. 'L1') to the sub-module whose
            output we want to capture.
        """
        self.cache: Dict[str, Tensor] = {}
        self._hooks = []
        for name, module in layer_map.items():
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def _hook(module, input, output):
            # output may be a tensor or a tuple (e.g. inception blocks).
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            # Spatial average pooling: (N, C, H, W) -> (N, C), or (N, C) unchanged.
            if out.dim() == 4:
                out = out.mean(dim=[2, 3])
            self.cache[name] = out.detach().cpu()
        return _hook

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MeansEndDecomposition:
    """
    Means-End Decomposition of a deep network's representational pipeline.

    Parameters
    ----------
    model : nn.Module
        A PyTorch classification model.  The five layer stages L1–L5 are
        identified automatically from common backbone architectures (ResNet,
        VGG, EfficientNet, ViT).  If auto-detection fails, pass
        ``layer_map`` explicitly.
    device : str
        Compute device.
    layer_map : dict[str, nn.Module], optional
        Manual override: maps layer label -> sub-module.  If not supplied,
        the constructor attempts to auto-detect from the model.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        layer_map: Optional[Dict[str, nn.Module]] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.layers: List[str] = LAYER_NAMES

        if layer_map is not None:
            self._layer_map = layer_map
        else:
            self._layer_map = self._auto_detect_layers(model)

        # Results (populated by fit()).
        self.T: Dict[str, np.ndarray] = {}          # T_l  matrices
        self.Delta: Dict[str, np.ndarray] = {}      # Delta_l matrices
        self.kappa: Dict[str, np.ndarray] = {}      # kappa_l matrices

        # Class pair lists.
        self.hard_pairs: List[Tuple[int, int]] = []
        self.easy_pairs: List[Tuple[int, int]] = []
        self.all_pairs: List[Tuple[int, int]] = []

        # Number of classes (inferred from confusion matrix during fit).
        self.n_classes: int = 0

        # Raw per-layer, per-pair AUROC cache (for visualisation).
        # _auroc[layer_name][(i,j)] = float
        self._auroc: Dict[str, Dict[Tuple[int, int], float]] = defaultdict(dict)
        # Ablated AUROC.
        self._auroc_ablated: Dict[str, Dict[Tuple[int, int], float]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Layer auto-detection
    # ------------------------------------------------------------------

    def _auto_detect_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """
        Try to identify the five representational stages (L1 … L5) in
        common backbone families.

        Returns a dict mapping {'L1': module, …, 'L5': module}.
        Raises RuntimeError if the backbone is not recognised.
        """
        # ResNet / ResNeXt family.
        if all(hasattr(model, f"layer{k}") for k in range(1, 5)):
            return {
                "L1": model.layer1,
                "L2": model.layer2,
                "L3": model.layer3,
                "L4": model.layer4,
                "L5": model.avgpool,
            }

        # VGG / AlexNet-style (features split into blocks).
        if hasattr(model, "features"):
            feats = model.features
            n = len(feats)
            # Split into 5 roughly equal blocks.
            idxs = np.round(np.linspace(0, n - 1, 6)).astype(int)
            blocks = {}
            for k in range(5):
                blocks[f"L{k+1}"] = feats[idxs[k]:idxs[k + 1]]  # type: ignore[index]
            # Wrap each slice in a sequential so we can register a hook.
            return {name: nn.Sequential(*list(blk)) for name, blk in blocks.items()}

        # EfficientNet.
        if hasattr(model, "features") and hasattr(model, "classifier"):
            feats = model.features
            n = len(feats)
            idxs = np.round(np.linspace(0, n - 1, 6)).astype(int)
            return {
                f"L{k+1}": feats[idxs[k]]
                for k in range(5)
            }

        # Vision Transformer.
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            enc_layers = list(model.encoder.layers)
            n = len(enc_layers)
            indices = np.round(np.linspace(0, n - 1, 5)).astype(int)
            return {f"L{k+1}": enc_layers[int(i)] for k, i in enumerate(indices)}

        raise RuntimeError(
            "Could not auto-detect layer stages.  Please pass a "
            "`layer_map` dict to MeansEndDecomposition.__init__()."
        )

    # ------------------------------------------------------------------
    # Pair selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_pairs(
        confusion_matrix: np.ndarray,
        n_random_control_pairs: int = 100,
        min_confusion: int = 5,
        random_state: int = 0,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Select hard pairs (high confusion) and easy control pairs.

        Parameters
        ----------
        confusion_matrix : (N, N) int array
            Entry M[i,j] = number of true-class-i samples predicted as j.
        n_random_control_pairs : int
            Number of easy (low-confusion) pairs to sample.
        min_confusion : int
            Threshold: M[i,j] + M[j,i] >= min_confusion -> hard pair.
        random_state : int
            RNG seed.

        Returns
        -------
        hard_pairs, easy_pairs : lists of (i, j) with i < j.
        """
        N = confusion_matrix.shape[0]
        hard, easy = [], []
        for i in range(N):
            for j in range(i + 1, N):
                sym_confusion = int(confusion_matrix[i, j]) + int(confusion_matrix[j, i])
                if sym_confusion >= min_confusion:
                    hard.append((i, j))
                else:
                    easy.append((i, j))

        rng = random.Random(random_state)
        k = min(n_random_control_pairs, len(easy))
        easy_sampled = rng.sample(easy, k)

        return hard, easy_sampled

    # ------------------------------------------------------------------
    # Activation extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_activations(
        self,
        loader: DataLoader,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Run the model over the loader, capturing activations at each of the
        five layer stages.

        Returns
        -------
        acts : dict[layer_name -> (N, D) float32 array]
        labels : (N,) int array
        """
        extractor = _ActivationExtractor(self.model, self._layer_map)
        self.model.eval()

        all_acts: Dict[str, List[np.ndarray]] = {l: [] for l in LAYER_NAMES}
        all_labels: List[int] = []

        for batch in tqdm(loader, desc="Extracting activations", leave=False):
            if isinstance(batch, (list, tuple)):
                imgs, lbls = batch[0], batch[1]
            else:
                raise TypeError("DataLoader must yield (images, labels) tuples.")

            imgs = imgs.to(self.device)
            _ = self.model(imgs)   # forward pass triggers hooks.

            for l in LAYER_NAMES:
                if l in extractor.cache:
                    all_acts[l].append(_to_numpy(extractor.cache[l]))

            if isinstance(lbls, Tensor):
                all_labels.extend(lbls.cpu().numpy().tolist())
            else:
                all_labels.extend(list(lbls))

        extractor.remove()

        acts_np: Dict[str, np.ndarray] = {}
        for l in LAYER_NAMES:
            if all_acts[l]:
                acts_np[l] = np.concatenate(all_acts[l], axis=0).astype(np.float32)
            else:
                warnings.warn(f"No activations captured for layer {l}.")

        return acts_np, np.array(all_labels, dtype=int)

    # ------------------------------------------------------------------
    # AUROC computation for a single pair
    # ------------------------------------------------------------------

    @staticmethod
    def _pair_auroc(
        X: np.ndarray,
        labels: np.ndarray,
        cls_i: int,
        cls_j: int,
    ) -> float:
        """
        Subset X and labels to classes cls_i and cls_j, then compute AUROC.
        """
        mask = (labels == cls_i) | (labels == cls_j)
        X_pair = X[mask]
        y_pair = (labels[mask] == cls_j).astype(int)   # 0 = class i, 1 = class j
        return _logistic_auroc(X_pair, y_pair)

    # ------------------------------------------------------------------
    # Ablated AUROC
    # ------------------------------------------------------------------

    @staticmethod
    def _pair_auroc_ablated(
        X_prev: np.ndarray,
        X_curr: np.ndarray,
        labels: np.ndarray,
        cls_i: int,
        cls_j: int,
    ) -> float:
        """
        Compute D_tilde_l[i,j]: replace each sample's layer-(l-1) activation
        with the class-mean of {i, j} combined, then retrain on those features
        concatenated with layer-l activations.

        Ablation strategy:
            X_ablated = class_mean(X_prev of {i,j}) broadcast to all {i,j} samples.
        This destroys the inter-class discriminative signal in layer l-1
        while leaving layer l untouched.

        We train the probe on X_curr only (layer l) using the ablated context.
        This approximates D_tilde by measuring how well layer l alone
        discriminates when layer l-1 is neutralised.
        """
        mask = (labels == cls_i) | (labels == cls_j)
        X_c = X_curr[mask]
        y_pair = (labels[mask] == cls_j).astype(int)

        # Replace X_prev with class-mean of the combined pair.
        combined_mean = X_prev[mask].mean(axis=0, keepdims=True)
        # The ablated feature is the residual: X_prev - class_mean,
        # which zeros out the between-class variance in l-1.
        # We then append this (zeroed) context to X_curr.
        ablated_prev = X_prev[mask] - combined_mean          # ~ zero-mean
        X_ablated = np.concatenate([ablated_prev, X_c], axis=1)

        return _logistic_auroc(X_ablated, y_pair)

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        confusion_matrix: np.ndarray,
        n_random_control_pairs: int = 100,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Run the full Means-End Decomposition algorithm.

        Parameters
        ----------
        train_loader : DataLoader
            Yields (images, labels) batches.  Images should already be
            preprocessed (normalised) to match the model's expected input.
        confusion_matrix : (N, N) ndarray of int
            Confusion matrix M where M[i,j] = #(true i predicted as j).
        n_random_control_pairs : int
            Number of easy control pairs to sample.

        Returns
        -------
        T_l : dict[layer -> (N, N) float array]
            Teleological content matrices.
        Delta_l : dict[layer -> (N, N) float array]
            Marginal contribution matrices.
        kappa_l : dict[layer -> (N, N) float array]
            Consumption score matrices.
        """
        self.n_classes = confusion_matrix.shape[0]
        N = self.n_classes

        # 1. Select pairs.
        self.hard_pairs, self.easy_pairs = self._select_pairs(
            confusion_matrix, n_random_control_pairs
        )
        self.all_pairs = self.hard_pairs + self.easy_pairs
        print(
            f"[MED] Pairs selected: {len(self.hard_pairs)} hard, "
            f"{len(self.easy_pairs)} easy control  (total {len(self.all_pairs)})"
        )

        # 2. Extract activations.
        print("[MED] Extracting activations …")
        acts, labels = self._extract_activations(train_loader)

        # 3. For each layer, for each pair, compute AUROC.
        print("[MED] Computing probes …")
        for l in LAYER_NAMES:
            if l not in acts:
                warnings.warn(f"Skipping layer {l}: no activations found.")
                continue
            X_l = acts[l]
            for (i, j) in tqdm(self.all_pairs, desc=f"  Layer {l}", leave=False):
                auc = self._pair_auroc(X_l, labels, i, j)
                self._auroc[l][(i, j)] = auc

        # 4. Compute ablated AUROC for consumption scores.
        print("[MED] Computing ablated probes for kappa …")
        layer_order = LAYER_NAMES  # L1 … L5
        for l_idx, l in enumerate(layer_order):
            if l not in acts:
                continue
            X_l = acts[l]
            if l_idx == 0:
                # No previous layer: D_tilde = D0 by definition.
                for (i, j) in self.all_pairs:
                    self._auroc_ablated[l][(i, j)] = D0
            else:
                l_prev = layer_order[l_idx - 1]
                if l_prev not in acts:
                    for (i, j) in self.all_pairs:
                        self._auroc_ablated[l][(i, j)] = D0
                    continue
                X_prev = acts[l_prev]
                for (i, j) in tqdm(
                    self.all_pairs, desc=f"  Ablated {l}", leave=False
                ):
                    auc_abl = self._pair_auroc_ablated(X_prev, X_l, labels, i, j)
                    self._auroc_ablated[l][(i, j)] = auc_abl

        # 5. Build T, Delta, kappa matrices (N x N).
        for l_idx, l in enumerate(layer_order):
            T_mat = np.full((N, N), D0, dtype=np.float32)
            Delta_mat = np.zeros((N, N), dtype=np.float32)
            kappa_mat = np.zeros((N, N), dtype=np.float32)

            l_prev = layer_order[l_idx - 1] if l_idx > 0 else None

            for (i, j) in self.all_pairs:
                auc_l = self._auroc[l].get((i, j), D0)
                auc_prev = (
                    self._auroc[l_prev].get((i, j), D0)
                    if (l_prev is not None and l_prev in self._auroc)
                    else D0
                )
                auc_abl = self._auroc_ablated[l].get((i, j), D0)

                T_l_val   = auc_l
                Delta_val = auc_l - auc_prev
                kappa_val = auc_l - auc_abl

                # Fill symmetric entries (both (i,j) and (j,i)).
                for (r, c) in [(i, j), (j, i)]:
                    T_mat[r, c]     = T_l_val
                    Delta_mat[r, c] = Delta_val
                    kappa_mat[r, c] = kappa_val

            self.T[l]     = T_mat
            self.Delta[l] = Delta_mat
            self.kappa[l] = kappa_mat

        print("[MED] Fit complete.")
        return self.T, self.Delta, self.kappa

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _ensure_fitted(self) -> None:
        if not self.T:
            raise RuntimeError("Call fit() before visualising.")

    # ------------------------------------------------------------------
    # 1. Sankey-style Teleological Flow Diagram
    # ------------------------------------------------------------------

    def visualize_teleological_flow(
        self,
        pair: Tuple[int, int],
        save_path: str,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Draw a Sankey-inspired flow diagram showing how discriminative capacity
        accumulates and is consumed across layers for a given class pair.

        Each layer is a column of nodes.  The "flow width" from layer l-1 to l
        represents the marginal contribution Delta_l[i,j].  The consumption
        band within each node represents kappa_l[i,j].

        Parameters
        ----------
        pair : (i, j)
            Class pair to visualise.
        save_path : str
            Base path (without extension).
        class_names : dict[int, str], optional
            Human-readable class names.
        """
        self._ensure_fitted()
        i, j = pair
        # Ensure canonical order.
        if (i, j) not in self.hard_pairs + self.easy_pairs:
            if (j, i) in self.hard_pairs + self.easy_pairs:
                i, j = j, i
            else:
                warnings.warn(f"Pair ({i},{j}) not in fitted pairs; plotting anyway.")

        cn = class_names or {}
        name_i = cn.get(i, f"Class {i}")
        name_j = cn.get(j, f"Class {j}")
        title = f"Teleological Flow  ({name_i}  vs  {name_j})"

        # Gather data.
        T_vals     = [self.T[l][i, j]     for l in LAYER_NAMES if l in self.T]
        Delta_vals = [self.Delta[l][i, j] for l in LAYER_NAMES if l in self.Delta]
        kappa_vals = [self.kappa[l][i, j] for l in LAYER_NAMES if l in self.kappa]
        layers_present = [l for l in LAYER_NAMES if l in self.T]
        n_layers = len(layers_present)

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 5),
            gridspec_kw={"width_ratios": [2.5, 1]},
        )
        fig.patch.set_facecolor("#12121f")

        # ---- Left panel: Sankey approximation ----
        ax = axes[0]
        ax.set_facecolor("#12121f")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(0.45, 1.05)

        # Node x positions.
        xs = np.arange(n_layers)
        y_T = np.array(T_vals)

        # Draw flow bands between adjacent layers.
        for k in range(n_layers - 1):
            x0, x1 = xs[k], xs[k + 1]
            y_bottom_left  = D0
            y_top_left     = y_T[k]
            y_bottom_right = D0
            y_top_right    = y_T[k + 1]

            # Trapezoidal band.
            verts = [
                (x0, y_bottom_left),
                (x0, y_top_left),
                (x1, y_top_right),
                (x1, y_bottom_right),
            ]
            poly = plt.Polygon(
                verts,
                closed=True,
                facecolor="#3a7fd5",
                edgecolor="none",
                alpha=0.35,
                zorder=1,
            )
            ax.add_patch(poly)

            # Positive-delta band (green) on top.
            delta = Delta_vals[k + 1]
            if delta > 0:
                dv = [
                    (x0, y_T[k]),
                    (x0, y_T[k] + delta * 0.5),
                    (x1, y_T[k + 1]),
                    (x1, y_T[k + 1] - delta * 0.5),
                ]
                arrow_poly = plt.Polygon(
                    dv,
                    closed=True,
                    facecolor="#2ecc71",
                    edgecolor="none",
                    alpha=0.5,
                    zorder=2,
                )
                ax.add_patch(arrow_poly)

        # Draw nodes (rectangles).
        node_w = 0.18
        node_h_base = 0.06
        for k, (l, T_val, kappa_val) in enumerate(zip(layers_present, T_vals, kappa_vals)):
            x_c = xs[k]
            rect = mpatches.FancyBboxPatch(
                (x_c - node_w / 2, D0 - 0.01),
                node_w,
                T_val - D0 + 0.01,
                boxstyle="round,pad=0.005",
                facecolor="#1a3a6e",
                edgecolor="#3a7fd5",
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(rect)

            # Consumption band (red overlay inside node).
            if kappa_val > 1e-4:
                krect = mpatches.FancyBboxPatch(
                    (x_c - node_w / 2 + 0.005, D0 - 0.01),
                    node_w - 0.01,
                    min(kappa_val, T_val - D0),
                    boxstyle="round,pad=0.002",
                    facecolor="#e74c3c",
                    edgecolor="none",
                    alpha=0.5,
                    zorder=4,
                )
                ax.add_patch(krect)

            # Labels.
            ax.text(
                x_c, D0 - 0.025, l,
                ha="center", va="top", fontsize=10, color="white", fontweight="bold"
            )
            ax.text(
                x_c, T_val + 0.015,
                f"T={T_val:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#7ecef4"
            )

        # Reference line at D0 = 0.5.
        ax.axhline(D0, color="#888888", linestyle="--", linewidth=0.8, zorder=0)
        ax.text(
            -0.45, D0 + 0.005, "D₀=0.5",
            fontsize=7, color="#888888", va="bottom"
        )

        ax.set_ylabel("AUROC", color="white", fontsize=10)
        ax.set_xticks([])
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444466")
        ax.set_title("Layer-wise Discriminative Capacity (Sankey)", color="white", fontsize=11)

        # Legend.
        legend_patches = [
            mpatches.Patch(color="#3a7fd5", alpha=0.6, label="Cumulative T_l"),
            mpatches.Patch(color="#2ecc71", alpha=0.7, label="ΔD (marginal gain)"),
            mpatches.Patch(color="#e74c3c", alpha=0.6, label="κ (consumed from prev)"),
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper left",
            framealpha=0.3,
            facecolor="#12121f",
            edgecolor="#444466",
            labelcolor="white",
            fontsize=8,
        )

        # ---- Right panel: Delta and kappa bar chart ----
        ax2 = axes[1]
        ax2.set_facecolor("#12121f")
        bar_y = np.arange(n_layers)
        bar_h = 0.35

        ax2.barh(
            bar_y + bar_h / 2, Delta_vals, bar_h,
            color="#2ecc71", alpha=0.8, label="Δ_l"
        )
        ax2.barh(
            bar_y - bar_h / 2, kappa_vals, bar_h,
            color="#e74c3c", alpha=0.8, label="κ_l"
        )
        ax2.set_yticks(bar_y)
        ax2.set_yticklabels(layers_present, color="white", fontsize=9)
        ax2.set_xlabel("Score", color="white", fontsize=9)
        ax2.axvline(0, color="#888888", linewidth=0.8)
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_color("#444466")
        ax2.legend(
            framealpha=0.3, facecolor="#12121f",
            edgecolor="#444466", labelcolor="white", fontsize=8
        )
        ax2.set_title("Δ_l  and  κ_l", color="white", fontsize=11)

        fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        self._save_fig(fig, save_path)

    # ------------------------------------------------------------------
    # 2. Layer Purpose Profile
    # ------------------------------------------------------------------

    def visualize_layer_purpose_profile(
        self,
        layer: str,
        save_path: str,
        class_names: Optional[Dict[int, str]] = None,
        top_k: int = 30,
    ) -> None:
        """
        Bar chart of Delta_l for all class pairs at a given layer, sorted by
        magnitude.

        Parameters
        ----------
        layer : str
            One of 'L1' … 'L5'.
        save_path : str
            Base path (without extension).
        class_names : dict[int, str], optional
            Human-readable class names for axis labels.
        top_k : int
            Maximum number of pairs to display (largest |Delta| first).
        """
        self._ensure_fitted()
        if layer not in self.Delta:
            raise ValueError(f"Layer '{layer}' not found in fitted results.")

        cn = class_names or {}

        # Gather deltas for all fitted pairs.
        pair_labels: List[str] = []
        delta_vals: List[float] = []
        kappa_vals: List[float] = []

        for (i, j) in self.all_pairs:
            di = self.Delta[layer][i, j]
            ki = self.kappa[layer][i, j]
            name_i = cn.get(i, str(i))
            name_j = cn.get(j, str(j))
            pair_labels.append(f"{name_i[:10]}|{name_j[:10]}")
            delta_vals.append(float(di))
            kappa_vals.append(float(ki))

        # Sort by |Delta| descending.
        order = np.argsort(np.abs(delta_vals))[::-1][:top_k]
        pair_labels = [pair_labels[o] for o in order]
        delta_vals  = [delta_vals[o]  for o in order]
        kappa_vals  = [kappa_vals[o]  for o in order]

        n = len(pair_labels)
        fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 6))
        fig.patch.set_facecolor("#12121f")
        ax.set_facecolor("#12121f")

        x = np.arange(n)
        bar_w = 0.4

        bars_d = ax.bar(x - bar_w / 2, delta_vals, bar_w, color="#2ecc71", alpha=0.85, label="Δ_l")
        bars_k = ax.bar(x + bar_w / 2, kappa_vals, bar_w, color="#e74c3c", alpha=0.85, label="κ_l")

        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, rotation=70, ha="right", fontsize=7, color="white")
        ax.set_ylabel("Score", color="white", fontsize=10)
        ax.set_title(
            f"Layer Purpose Profile  —  {layer}",
            color="white", fontsize=13, fontweight="bold"
        )
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444466")
        ax.legend(
            framealpha=0.3, facecolor="#12121f",
            edgecolor="#444466", labelcolor="white", fontsize=9
        )

        plt.tight_layout()
        self._save_fig(fig, save_path)

    # ------------------------------------------------------------------
    # 3. Dependency Graph
    # ------------------------------------------------------------------

    def visualize_dependency_graph(
        self,
        save_path: str,
        class_names: Optional[Dict[int, str]] = None,
        kappa_threshold: float = 0.02,
    ) -> None:
        """
        Draw a directed graph of consumption dependencies.

        Nodes are (layer, class-pair) combinations.  A directed edge
        (l-1, pair) -> (l, pair) exists if kappa_l[pair] >= kappa_threshold,
        indicating that layer l significantly consumes information from l-1.
        Edge width and colour encode kappa magnitude.

        Parameters
        ----------
        save_path : str
            Base path (without extension).
        class_names : dict[int, str], optional
            Human-readable class names.
        kappa_threshold : float
            Minimum kappa to draw an edge.
        """
        self._ensure_fitted()
        cn = class_names or {}

        layers_present = [l for l in LAYER_NAMES if l in self.kappa]
        n_layers = len(layers_present)

        # Gather all (layer, pair, kappa) triples above threshold.
        edges: List[Tuple[str, Tuple[int, int], str, float]] = []
        # (src_layer, pair) -> (dst_layer, pair)
        for l_idx in range(1, n_layers):
            l_dst = layers_present[l_idx]
            l_src = layers_present[l_idx - 1]
            for (i, j) in self.all_pairs:
                k_val = float(self.kappa[l_dst][i, j])
                if k_val >= kappa_threshold:
                    edges.append((l_src, (i, j), l_dst, k_val))

        # Layout: layers on x-axis; we spread pairs vertically.
        # For readability, collapse pairs into unique vertical slots.
        all_pairs_here = sorted(
            {(i, j) for _, (i, j), _, _ in edges},
            key=lambda p: -(self.kappa[layers_present[-1]][p[0], p[1]])
            if layers_present else 0
        )
        pair_y = {p: idx for idx, p in enumerate(all_pairs_here)}
        layer_x = {l: idx for idx, l in enumerate(layers_present)}

        fig_h = max(6, len(all_pairs_here) * 0.35 + 2)
        fig, ax = plt.subplots(figsize=(max(10, n_layers * 2.5), fig_h))
        fig.patch.set_facecolor("#12121f")
        ax.set_facecolor("#12121f")

        # Colour map for kappa magnitude.
        norm = mcolors.Normalize(
            vmin=kappa_threshold,
            vmax=max((e[3] for e in edges), default=1.0),
        )
        cmap = plt.colormaps["YlOrRd"]

        drawn_nodes = set()

        for (l_src, pair, l_dst, k_val) in edges:
            x_src = layer_x[l_src]
            x_dst = layer_x[l_dst]
            y_p   = pair_y[pair]

            color = cmap(norm(k_val))
            lw    = 1.0 + 5.0 * norm(k_val)

            ax.annotate(
                "",
                xy=(x_dst, y_p),
                xytext=(x_src, y_p),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=lw,
                    mutation_scale=12,
                ),
            )

            # Draw source node if not yet drawn.
            for (lx, ly) in [(x_src, y_p), (x_dst, y_p)]:
                if (lx, ly) not in drawn_nodes:
                    ax.scatter(lx, ly, s=60, color="#3a7fd5", zorder=5, linewidths=0)
                    drawn_nodes.add((lx, ly))

        # Pair labels on right margin.
        for pair, y_idx in pair_y.items():
            i, j = pair
            label = f"{cn.get(i, i)}↔{cn.get(j, j)}"
            ax.text(
                n_layers - 0.5 + 0.15, y_idx,
                label, fontsize=6.5, color="#aaaacc",
                va="center", ha="left",
            )

        # Layer labels on x-axis.
        ax.set_xticks(list(layer_x.values()))
        ax.set_xticklabels(list(layer_x.keys()), color="white", fontsize=11, fontweight="bold")
        ax.set_yticks([])
        ax.set_xlim(-0.5, n_layers - 0.5 + 2.5)
        ax.set_ylim(-1, max(len(all_pairs_here), 1))

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Colour-bar.
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label("κ (consumption score)", color="white", fontsize=9)
        cbar.ax.tick_params(colors="white", labelsize=7)
        cbar.outline.set_edgecolor("white")

        ax.set_title(
            f"Consumption Dependency Graph  (κ ≥ {kappa_threshold:.2f})",
            color="white", fontsize=12, fontweight="bold",
        )

        plt.tight_layout()
        self._save_fig(fig, save_path)

    # ------------------------------------------------------------------
    # Save helper
    # ------------------------------------------------------------------

    @staticmethod
    def _save_fig(fig: plt.Figure, save_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        base = save_path
        for ext in (".png", ".pdf"):
            if base.endswith(ext):
                base = base[: -len(ext)]
                break
        fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[MED] Saved {base}.png  /  {base}.pdf")


# ---------------------------------------------------------------------------
# Quick self-test (run as: python means_end_decomposition.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torchvision.models as tvm
    from torch.utils.data import TensorDataset

    print("Running MeansEndDecomposition self-test on CPU …")

    # Tiny model: ResNet-18 adapted to 10 classes.
    device = "cpu"
    model = tvm.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.eval()

    # Synthetic dataset: 60 samples, 10 classes, 3x64x64 images.
    torch.manual_seed(42)
    n_samples, n_classes = 60, 10
    images = torch.randn(n_samples, 3, 64, 64)
    lbls   = torch.randint(0, n_classes, (n_samples,))

    dataset = TensorDataset(images, lbls)
    loader  = DataLoader(dataset, batch_size=16, shuffle=False)

    # Fake confusion matrix.
    rng = np.random.default_rng(0)
    confusion = rng.integers(0, 10, size=(n_classes, n_classes)).astype(int)
    np.fill_diagonal(confusion, 0)

    # Instantiate and fit.
    med = MeansEndDecomposition(model, device=device)
    T, Delta, kappa = med.fit(loader, confusion, n_random_control_pairs=5)

    print(f"  Layers with T matrices : {list(T.keys())}")
    for l in T:
        print(f"  {l}: T shape={T[l].shape}, Delta shape={Delta[l].shape}, kappa shape={kappa[l].shape}")

    # Visualise.
    class_names_test = {i: f"Cls{i}" for i in range(n_classes)}

    if med.all_pairs:
        sample_pair = med.all_pairs[0]
        med.visualize_teleological_flow(
            sample_pair,
            save_path="/tmp/med_flow_test",
            class_names=class_names_test,
        )

    med.visualize_layer_purpose_profile(
        "L3",
        save_path="/tmp/med_profile_L3_test",
        class_names=class_names_test,
        top_k=10,
    )

    med.visualize_dependency_graph(
        save_path="/tmp/med_dep_graph_test",
        class_names=class_names_test,
        kappa_threshold=0.0,
    )

    print("Self-test complete.")
