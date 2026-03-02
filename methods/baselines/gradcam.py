"""
GradCAM baseline explainer using Captum's LayerGradCam.

Targets layer4[-1] of a ResNet-50 model (14x14 feature map) and
upsamples to the full (448, 448) input resolution.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from captum.attr import LayerGradCam


class GradCAMExplainer:
    """
    Computes GradCAM attributions at layer4[-1] of a ResNet-50 model.

    The raw GradCAM output is a (1, 1, 14, 14) tensor that is ReLU-ed,
    normalised to [0, 1], and bilinearly upsampled to (448, 448).

    Parameters
    ----------
    model : nn.Module
        A ResNet-50 (torchvision or compatible) with attributes
        ``layer4`` (nn.Sequential of Bottleneck blocks).
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        # Target the last bottleneck block of layer4
        target_layer = model.layer4[-1]  # type: ignore[index]
        self.layer_gc = LayerGradCam(self.model, target_layer)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        x: torch.Tensor,
        target_class: int,
        image_size: int = 448,
    ) -> np.ndarray:
        """
        Compute a GradCAM heatmap for a single input image.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(1, 3, H, W)`` or ``(3, H, W)``.
            Will be moved to ``self.device``.
        target_class : int
            The class index to attribute.
        image_size : int
            Side length of the output heatmap (default 448).

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(image_size, image_size)`` in [0, 1].
        """
        # Ensure batch dimension and correct device
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        # Captum expects requires_grad for some methods; ensure it
        x = x.requires_grad_(True)

        # Compute GradCAM attribution
        # Returns shape (1, 1, H_l, W_l) where H_l = W_l = 14 for layer4
        attr = self.layer_gc.attribute(
            x,
            target=target_class,
            relu_attributions=True,
        )

        # Upsample to target image size
        attr_up = F.interpolate(
            attr,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, 448, 448)

        # Convert to numpy and normalise
        heatmap = attr_up.squeeze().detach().cpu().float().numpy()
        heatmap = np.maximum(heatmap, 0.0)  # ReLU (already done, but safety)

        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.astype(np.float32)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(
        self,
        x: torch.Tensor,
        heatmap: np.ndarray,
        class_name: str,
        save_path: str,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> None:
        """
        Overlay the GradCAM heatmap on the input image and save the figure.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(1, 3, H, W)`` or ``(3, H, W)`` in ImageNet
            normalised form.
        heatmap : np.ndarray
            Float32 heatmap of shape ``(H, W)`` in [0, 1], typically from
            ``self.compute()``.
        class_name : str
            Human-readable class name for the figure title.
        save_path : str
            Absolute path where the PNG will be written.
        alpha : float
            Opacity of the heatmap overlay (default 0.4).
        colormap : str
            Matplotlib colormap name (default 'jet').
        """
        if x.dim() == 4:
            x = x.squeeze(0)

        # Denormalise: reverse ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_np = (x.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        # Apply colormap to heatmap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[..., :3]  # (H, W, 3), drop alpha

        # Resize heatmap to match image if necessary
        if heatmap.shape != img_np.shape[:2]:
            from PIL import Image
            hm_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
            hm_pil = hm_pil.resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)
            heatmap_resized = np.array(hm_pil) / 255.0
            heatmap_colored = cmap(heatmap_resized)[..., :3]

        overlay = (1 - alpha) * img_np + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap=colormap)
        axes[1].set_title("GradCAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay — {class_name}")
        axes[2].axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
