"""
Integrated Gradients baseline explainer using Captum.

Uses a black (zero) baseline and 50 interpolation steps.  The
per-channel attribution tensor is summed over the channel dimension
and the absolute value is taken, yielding a single-channel saliency
map that is then upsampled to (448, 448).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from captum.attr import IntegratedGradients


class IGExplainer:
    """
    Integrated Gradients explainer for image classification models.

    Uses a black (all-zeros) baseline and 50 interpolation steps by
    default.  Attribution values are summed over the colour channel
    axis and the result is upsampled to ``image_size x image_size``.

    Parameters
    ----------
    model : nn.Module
        A classification model (e.g. ResNet-50).  Must produce
        un-normalised logits.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.ig = IntegratedGradients(self.model)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        x: torch.Tensor,
        target_class: int,
        n_steps: int = 50,
        image_size: int = 448,
    ) -> np.ndarray:
        """
        Compute an Integrated Gradients saliency map.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(1, 3, H, W)`` or ``(3, H, W)``.
        target_class : int
            The class index to attribute toward.
        n_steps : int
            Number of IG interpolation steps (default 50).
        image_size : int
            Output spatial size; input is upsampled if necessary.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(image_size, image_size)`` representing
            the channel-summed, absolute-value attribution, normalised to
            [0, 1].
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device).requires_grad_(True)

        # Black baseline: all zeros, same shape as input
        baseline = torch.zeros_like(x)

        # Compute IG attributions
        # Output shape: (1, 3, H, W)
        attrs = self.ig.attribute(
            x,
            baselines=baseline,
            target=target_class,
            n_steps=n_steps,
            return_convergence_delta=False,
        )

        # Sum over channels and take absolute value -> (H, W)
        attr_map = attrs.sum(dim=1).abs()  # (1, H, W)

        # Upsample to target size
        attr_map = F.interpolate(
            attr_map.unsqueeze(1),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # (image_size, image_size)

        attr_np = attr_map.detach().cpu().float().numpy()

        # Normalise to [0, 1]
        max_val = attr_np.max()
        if max_val > 0:
            attr_np = attr_np / max_val

        return attr_np.astype(np.float32)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(
        self,
        x: torch.Tensor,
        attr_map: np.ndarray,
        class_name: str,
        save_path: str,
        alpha: float = 0.4,
        colormap: str = "hot",
        positive_only: bool = True,
    ) -> None:
        """
        Overlay the IG attribution map on the input image and save.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(1, 3, H, W)`` or ``(3, H, W)`` in ImageNet
            normalised form.
        attr_map : np.ndarray
            Attribution map of shape ``(H, W)`` in [0, 1], from
            ``self.compute()``.
        class_name : str
            Human-readable label for the figure title.
        save_path : str
            Destination PNG path.
        alpha : float
            Heatmap overlay opacity (default 0.4).
        colormap : str
            Matplotlib colormap (default 'hot').
        positive_only : bool
            If True, only positive attributions are shown.
        """
        if x.dim() == 4:
            x = x.squeeze(0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_np = (x.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        display_map = np.maximum(attr_map, 0.0) if positive_only else attr_map

        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(display_map)[..., :3]

        overlay = (1 - alpha) * img_np + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        im = axes[1].imshow(display_map, cmap=colormap)
        axes[1].set_title("IG Attribution Map")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay — {class_name}")
        axes[2].axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
