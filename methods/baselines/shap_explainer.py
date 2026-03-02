"""
SHAP baseline explainer using Captum's GradientShap.

GradientShap selects a random baseline from the provided background
dataset for each sample, then runs Integrated Gradients between that
baseline and the input.  The resulting per-pixel, per-channel SHAP
values are summed over the channel axis and upsampled to (448, 448).
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from captum.attr import GradientShap


class SHAPExplainer:
    """
    GradientShap (Captum) explainer for image classification.

    A random subset of ``n_background`` training images is collected
    as the background distribution.  For each query, GradientShap
    draws a random baseline from this set, runs IG, and returns
    (channel-summed) SHAP values normalised to [0, 1].

    Parameters
    ----------
    model : nn.Module
        Classification model (e.g. ResNet-50) returning logits.
    background_dataset : Dataset or DataLoader or torch.Tensor
        Source of background images.  Accepts:

        - A PyTorch ``Dataset`` whose ``__getitem__`` returns
          ``(image_tensor, label)`` or just ``image_tensor``.
        - A pre-built ``DataLoader``.
        - A ``torch.Tensor`` of shape ``(N, 3, H, W)``.

    n_background : int
        Number of background images to collect (default 200).
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model: nn.Module,
        background_dataset: Union[Dataset, DataLoader, torch.Tensor],
        n_background: int = 200,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.n_background = n_background
        self.background = self._collect_background(background_dataset, n_background)
        self.gradient_shap = GradientShap(self.model)

    # ------------------------------------------------------------------
    # Background collection
    # ------------------------------------------------------------------

    def _collect_background(
        self,
        source: Union[Dataset, DataLoader, torch.Tensor],
        n: int,
    ) -> torch.Tensor:
        """
        Collect up to *n* images from *source* as the background tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(min(n, N_available), 3, H, W)`` on ``self.device``.
        """
        if isinstance(source, torch.Tensor):
            bg = source[:n].float().to(self.device)
            return bg

        # Collect from Dataset or DataLoader
        images = []
        if isinstance(source, Dataset):
            # Wrap in a DataLoader for batched iteration
            loader = DataLoader(source, batch_size=32, shuffle=False, num_workers=0)
        else:
            loader = source

        for batch in loader:
            # Support (image, label) tuples or bare image tensors
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch

            images.append(imgs.float())
            if sum(b.shape[0] for b in images) >= n:
                break

        bg = torch.cat(images, dim=0)[:n].to(self.device)
        return bg

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        x: torch.Tensor,
        target_class: int,
        n_samples: int = 5,
        image_size: int = 448,
        stdevs: float = 0.09,
    ) -> np.ndarray:
        """
        Compute GradientShap attributions for a single input image.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(1, 3, H, W)`` or ``(3, H, W)``.
        target_class : int
            Target class index.
        n_samples : int
            Number of random baselines to average over (passed to
            ``GradientShap.attribute`` as ``n_samples``).
        image_size : int
            Output spatial size (default 448).
        stdevs : float
            Noise standard deviation added to baselines (default 0.09).

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(image_size, image_size)`` in [0, 1].
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device).requires_grad_(True)

        # GradientShap requires the baseline to have the same batch dim
        # as the input (or be broadcastable).  We pass the full background
        # and let Captum randomly sample from it.
        attr = self.gradient_shap.attribute(
            x,
            baselines=self.background,
            target=target_class,
            n_samples=n_samples,
            stdevs=stdevs,
        )  # (1, 3, H, W)

        # Sum over channels and take absolute value
        shap_map = attr.sum(dim=1).abs()  # (1, H, W)

        # Upsample
        shap_map = F.interpolate(
            shap_map.unsqueeze(1),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # (image_size, image_size)

        shap_np = shap_map.detach().cpu().float().numpy()

        max_val = shap_np.max()
        if max_val > 0:
            shap_np = shap_np / max_val

        return shap_np.astype(np.float32)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(
        self,
        x: torch.Tensor,
        shap_map: np.ndarray,
        class_name: str,
        save_path: str,
        alpha: float = 0.4,
        colormap: str = "RdBu_r",
    ) -> None:
        """
        Overlay the SHAP map on the input image and save the figure.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(1, 3, H, W)`` or ``(3, H, W)`` in ImageNet
            normalised form.
        shap_map : np.ndarray
            Attribution map of shape ``(H, W)`` in [0, 1].
        class_name : str
            Human-readable class name for the title.
        save_path : str
            Destination PNG path.
        alpha : float
            Overlay opacity (default 0.4).
        colormap : str
            Matplotlib colormap (default 'RdBu_r').
        """
        if x.dim() == 4:
            x = x.squeeze(0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_np = (x.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(shap_map)[..., :3]

        overlay = (1 - alpha) * img_np + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        im = axes[1].imshow(shap_map, cmap=colormap, vmin=0, vmax=1)
        axes[1].set_title("SHAP Map")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay — {class_name}")
        axes[2].axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
