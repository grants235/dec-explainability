"""
Purposive Saliency Maps for Teleological Explainability.

For an input x (shape 1x3x448x448), true class c, and a confusion set C(x)
with associated weights {w_j}, this module computes per-competitor saliency
maps via Integrated Gradients along the margin logit

    m_j(x) = f_c(x) - f_j(x)

using a zero (black) baseline, with N=50 midpoint-rule quadrature steps.

The aggregated map is:

    S_agg(u,v) = sum_j  w_j * |S_j(u,v)|

and the annotation map assigns each pixel to its most salient competitor:

    A(u,v) = argmax_j |S_j(u,v)|

Reference: Shanklin et al., "Teleological Explainability" (2026).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_numpy(t: Tensor) -> np.ndarray:
    """Detach, move to CPU, and convert a tensor to a numpy array."""
    return t.detach().cpu().float().numpy()


def _channel_sum_abs(t: Tensor) -> Tensor:
    """Sum the absolute value of a (C, H, W) tensor along the channel axis -> (H, W)."""
    return t.abs().sum(dim=0)


def _normalise_map(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise a 2-D array to [0, 1].  Returns array of zeros if flat."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _unnormalise_image(x: Tensor) -> np.ndarray:
    """
    Convert a (1, 3, H, W) normalised tensor to a (H, W, 3) uint8 numpy array
    suitable for imshow.

    Assumes ImageNet normalisation:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    img = x * std + mean                     # (1, 3, H, W) in [0, 1]
    img = img.squeeze(0).permute(1, 2, 0)    # (H, W, 3)
    img = img.clamp(0.0, 1.0)
    return _to_numpy(img)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class PurposiveSaliency:
    """
    Compute Purposive Saliency Maps using Integrated Gradients along the
    margin logit for each competitor in a confusion set.

    Parameters
    ----------
    model : nn.Module
        A PyTorch classification model whose forward pass returns logits of
        shape (batch, num_classes).
    device : str
        Target device ('cuda' or 'cpu').
    """

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        self.model = model.to(device)
        self.device = device
        # Zero-baseline corresponds to a black image *before* normalisation.
        # After standard ImageNet normalisation the pixel values of a black image
        # (all zeros in [0,255]) become (-mean/std), but the specification calls for
        # x_bar = zeros in tensor space (i.e., the all-zero tensor after whatever
        # preprocessing the caller applies).  We therefore use a zero tensor.
        self._baseline_value: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        x: Tensor,
        true_class: int,
        confusion_set: List[int],
        weights: Dict[int, float],
        n_steps: int = 50,
    ) -> Tuple[Dict[int, Tensor], Tensor, Tensor]:
        """
        Compute Purposive Saliency Maps.

        Parameters
        ----------
        x : Tensor
            Input image of shape (1, 3, 448, 448).
        true_class : int
            Index of the ground-truth class c.
        confusion_set : list[int]
            Ordered list of competitor class indices j.
        weights : dict[int, float]
            Per-competitor weights w_j  (need not be normalised).
        n_steps : int
            Number of midpoint-rule integration steps (default 50).

        Returns
        -------
        per_competitor_maps : dict[int, Tensor]
            For each competitor j: channel-summed, element-wise-product saliency
            map of shape (H, W) = (448, 448).  Values may be negative.
        s_agg : Tensor
            Weighted aggregated saliency (H, W), non-negative.
        annotation_map : LongTensor
            Per-pixel argmax competitor index (H, W); values are class indices
            from confusion_set.
        """
        assert x.dim() == 4 and x.shape[0] == 1, (
            f"Expected x of shape (1, C, H, W), got {tuple(x.shape)}"
        )
        if len(confusion_set) == 0:
            raise ValueError("confusion_set must be non-empty.")

        x = x.to(self.device)

        # Baseline: zero tensor in the same space as x.
        x_bar = torch.zeros_like(x, device=self.device)

        # Difference vector shared across all steps.
        delta = x - x_bar                          # (1, 3, H, W)

        # Accumulate per-competitor integrated gradients.
        # integrated_grads[j] accumulates sum_k grad_x' m_j(x_bar + t_k * delta).
        J = len(confusion_set)
        H, W = x.shape[2], x.shape[3]

        # (J, 3, H, W) float32 accumulator on device.
        accumulated: Tensor = torch.zeros(J, 3, H, W, device=self.device, dtype=torch.float32)

        self.model.eval()

        for k in range(n_steps):
            # Midpoint rule: t_k = (k + 0.5) / n_steps  for k in {0, …, N-1}.
            t_k = (k + 0.5) / n_steps
            x_interp = x_bar + t_k * delta          # (1, 3, H, W)
            x_interp = x_interp.detach().requires_grad_(True)

            # Single forward pass -> all logits.
            logits = self.model(x_interp)            # (1, num_classes)

            # Build all margin outputs in one shot.
            margins: List[Tensor] = [
                logits[0, true_class] - logits[0, j]
                for j in confusion_set
            ]

            # Compute gradients for all margins simultaneously.
            # Each element of grads is d/dx' m_j, shape (1, 3, H, W).
            grads: Tuple[Tensor, ...] = torch.autograd.grad(
                outputs=margins,
                inputs=x_interp,
                grad_outputs=[torch.ones((), device=self.device)] * J,
                retain_graph=False,
                create_graph=False,
            )

            for idx, g in enumerate(grads):
                # g: (1, 3, H, W)
                accumulated[idx] += g.squeeze(0)    # -> (3, H, W)

        # Midpoint rule approximation: multiply by step size.
        accumulated = accumulated / n_steps          # (J, 3, H, W)

        # Element-wise product with delta (the input-baseline difference).
        # delta: (1, 3, H, W) -> broadcast over J.
        delta_np = delta.squeeze(0)                  # (3, H, W)
        # S_j[channel] = delta[channel] * integral_j[channel]  (J, 3, H, W)
        saliency_volume = accumulated * delta_np.unsqueeze(0)  # (J, 3, H, W)

        # Per-competitor maps: sum absolute value over channels -> (J, H, W).
        # NOTE: we keep the signed version for per_competitor_maps as (H,W)
        # tensors where the sign indicates whether eliminating the margin component
        # helps or hurts.  The specification says S_j = (x - x_bar) ⊙ S_j,
        # so we report the signed channel-summed map.
        per_competitor_maps: Dict[int, Tensor] = {}
        signed_maps: List[Tensor] = []
        for idx, j in enumerate(confusion_set):
            # Sum over channels (signed), shape (H, W).
            s_j_signed = saliency_volume[idx].sum(dim=0)   # (H, W)
            per_competitor_maps[j] = s_j_signed.detach()
            signed_maps.append(s_j_signed)

        # Aggregated map: S_agg = sum_j w_j * |S_j|, shape (H, W).
        s_agg = torch.zeros(H, W, device=self.device, dtype=torch.float32)
        for idx, j in enumerate(confusion_set):
            w_j = float(weights.get(j, 1.0))
            s_agg += w_j * signed_maps[idx].abs()

        # Annotation map: argmax_j |S_j(u,v)| -> class index, shape (H, W).
        # Stack |S_j| maps: (J, H, W).
        abs_stack = torch.stack([signed_maps[idx].abs() for idx in range(J)], dim=0)
        argmax_idx = abs_stack.argmax(dim=0)          # (H, W) in {0, …, J-1}
        # Map argmax integer index back to class index.
        class_index_tensor = torch.tensor(confusion_set, device=self.device)
        annotation_map = class_index_tensor[argmax_idx]  # (H, W) LongTensor

        return per_competitor_maps, s_agg.detach(), annotation_map.detach()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize(
    x: Tensor,
    per_competitor_maps: Dict[int, Tensor],
    s_agg: Tensor,
    annotation_map: Tensor,
    class_names: Dict[int, str],
    confusion_set: List[int],
    save_path: str,
) -> None:
    """
    Visualise Purposive Saliency results and save as both .png and .pdf.

    Layout
    ------
    Row 0 (top grid):
        [Original image] | [S_agg overlay] | [S_j heatmap for each competitor]

    Row 1 (bottom):
        [Annotation map spanning full width, with colour legend]

    Parameters
    ----------
    x : Tensor
        Input image of shape (1, 3, H, W).
    per_competitor_maps : dict[int, Tensor]
        Output of PurposiveSaliency.compute().
    s_agg : Tensor
        Aggregated saliency map (H, W).
    annotation_map : Tensor
        Per-pixel competitor class index (H, W) LongTensor.
    class_names : dict[int, str]
        Maps class index -> human-readable name string.
    confusion_set : list[int]
        Ordered list of competitor class indices (determines column order).
    save_path : str
        Base path (without extension).  Files are saved as
        ``<save_path>.png`` and ``<save_path>.pdf``.
    """
    J = len(confusion_set)

    # ------------------------------------------------------------------
    # Build colour palette for competitors (one colour per competitor).
    # ------------------------------------------------------------------
    palette = plt.get_cmap("tab10")
    competitor_colors: Dict[int, np.ndarray] = {
        j: np.array(palette(idx % 10)[:3])
        for idx, j in enumerate(confusion_set)
    }

    # ------------------------------------------------------------------
    # Prepare arrays.
    # ------------------------------------------------------------------
    img_np = _unnormalise_image(x)                        # (H, W, 3)
    s_agg_np = _normalise_map(_to_numpy(s_agg))           # (H, W) in [0,1]
    ann_np = _to_numpy(annotation_map).astype(int)        # (H, W) int

    H, W = s_agg_np.shape

    # Per-competitor normalised absolute saliency.
    comp_maps_np: Dict[int, np.ndarray] = {
        j: _normalise_map(_to_numpy(per_competitor_maps[j].abs()))
        for j in confusion_set
    }

    # Colour overlay for annotation map.
    ann_rgb = np.zeros((H, W, 3), dtype=np.float32)
    for j in confusion_set:
        mask = ann_np == j
        ann_rgb[mask] = competitor_colors[j]

    # ------------------------------------------------------------------
    # Figure geometry:
    #   Top row:  1 (original) + 1 (agg) + J (per-competitor) columns.
    #   Bottom row: 1 column spanning everything (annotation map).
    # ------------------------------------------------------------------
    n_top_cols = 2 + J
    fig_w = max(4 * n_top_cols, 10)
    fig_h = 10  # two rows

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    fig.patch.set_facecolor("#1a1a2e")

    # GridSpec: 2 rows.  Row 0 has n_top_cols columns; row 1 spans all.
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    gs_outer = GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[1.6, 1.0],
        hspace=0.35,
        left=0.04, right=0.98, top=0.93, bottom=0.05,
    )
    gs_top = GridSpecFromSubplotSpec(1, n_top_cols, subplot_spec=gs_outer[0], wspace=0.08)
    ax_bottom = fig.add_subplot(gs_outer[1])

    # Helper: shared axis style.
    def _style_ax(ax: plt.Axes, title: str) -> None:
        ax.set_title(title, fontsize=8, color="white", pad=3)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # -- Original image --------------------------------------------------
    ax_orig = fig.add_subplot(gs_top[0])
    ax_orig.imshow(img_np)
    _style_ax(ax_orig, "Original")

    # -- Aggregated saliency overlay -------------------------------------
    ax_agg = fig.add_subplot(gs_top[1])
    ax_agg.imshow(img_np)
    agg_rgba = cm.inferno(s_agg_np)                 # (H, W, 4)
    agg_rgba[..., 3] = s_agg_np * 0.75              # alpha proportional to intensity
    ax_agg.imshow(agg_rgba, interpolation="bilinear")
    _style_ax(ax_agg, "S_agg (weighted)")

    # Colour-bar for aggregated map (thin, placed to the right of ax_agg).
    sm_agg = cm.ScalarMappable(cmap="inferno", norm=mcolors.Normalize(0, 1))
    sm_agg.set_array([])
    cbar_agg = fig.colorbar(sm_agg, ax=ax_agg, fraction=0.046, pad=0.04, orientation="vertical")
    cbar_agg.ax.tick_params(colors="white", labelsize=6)
    cbar_agg.outline.set_edgecolor("white")

    # -- Per-competitor heatmaps -----------------------------------------
    for col_idx, j in enumerate(confusion_set):
        ax_j = fig.add_subplot(gs_top[2 + col_idx])
        cmap_j = "RdBu_r"
        # Show signed map normalised to [-1, 1] for directional information.
        s_j_np = _to_numpy(per_competitor_maps[j])
        v_abs = max(np.abs(s_j_np).max(), 1e-12)
        ax_j.imshow(
            s_j_np,
            cmap=cmap_j,
            vmin=-v_abs,
            vmax=v_abs,
            interpolation="bilinear",
        )
        name = class_names.get(j, f"class {j}")
        _style_ax(ax_j, f"S_j: {name}")

    # -- Annotation map --------------------------------------------------
    ax_bottom.imshow(ann_rgb, interpolation="nearest")
    ax_bottom.set_title(
        "Annotation Map  A(u,v) = argmax_j |S_j(u,v)|",
        fontsize=9,
        color="white",
        pad=5,
    )
    ax_bottom.axis("off")

    # Legend patches.
    patches = [
        mpatches.Patch(
            color=competitor_colors[j],
            label=class_names.get(j, f"class {j}"),
        )
        for j in confusion_set
    ]
    legend = ax_bottom.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(J, 6),
        frameon=True,
        framealpha=0.4,
        facecolor="#1a1a2e",
        edgecolor="white",
        fontsize=8,
        labelcolor="white",
    )

    # Global title.
    fig.suptitle(
        "Purposive Saliency Maps",
        fontsize=13,
        color="white",
        fontweight="bold",
        y=0.98,
    )

    # Set all axes backgrounds to near-black to match dark theme.
    for ax in fig.axes:
        ax.set_facecolor("#0d0d1a")

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    base = save_path if not save_path.endswith((".png", ".pdf")) else save_path.rsplit(".", 1)[0]

    fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PurposiveSaliency] Saved visualisation to {base}.png  /  {base}.pdf")


# ---------------------------------------------------------------------------
# Quick self-test (run as: python purposive_saliency.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torchvision.models as tvm

    print("Running PurposiveSaliency self-test on CPU with a tiny ResNet-18 stub …")

    device = "cpu"
    model = tvm.resnet18(weights=None)

    # Adapt the final layer to 10 classes for speed.
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.eval()

    # Dummy input.
    torch.manual_seed(0)
    x_test = torch.randn(1, 3, 224, 224)

    ps = PurposiveSaliency(model, device=device)

    true_cls = 0
    conf_set = [1, 2, 3]
    wts = {1: 0.5, 2: 0.3, 3: 0.2}

    maps, s_agg, ann = ps.compute(x_test, true_cls, conf_set, wts, n_steps=5)

    print(f"  per_competitor_maps keys : {list(maps.keys())}")
    for k, v in maps.items():
        print(f"    class {k}: shape={tuple(v.shape)}, min={v.min():.4f}, max={v.max():.4f}")
    print(f"  s_agg shape={tuple(s_agg.shape)}, min={s_agg.min():.4f}, max={s_agg.max():.4f}")
    print(f"  annotation_map shape={tuple(ann.shape)}, unique values={ann.unique().tolist()}")

    class_names_test = {i: f"Bird_{i}" for i in range(10)}
    visualize(
        x_test,
        maps,
        s_agg,
        ann,
        class_names_test,
        conf_set,
        save_path="/tmp/purposive_saliency_test",
    )
    print("Self-test complete.")
