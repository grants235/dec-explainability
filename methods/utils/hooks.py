"""
Activation extraction hooks for ResNet-50.

Registers forward hooks on:
  L1: layer1[-1].relu  -> (B, 256, 112, 112)
  L2: layer2[-1].relu  -> (B, 512,  56,  56)
  L3: layer3[-1].relu  -> (B,1024,  28,  28)
  L4: layer4[-1].relu  -> (B,2048,  14,  14)
  L5: avgpool (pre-FC) -> (B,2048,   1,   1)

Spatial average pooling over L1-L4 yields 1D feature vectors used by the
means-end probes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet


# ---------------------------------------------------------------------------
# Layer-ID constants
# ---------------------------------------------------------------------------
LAYER_IDS: List[str] = ["L1", "L2", "L3", "L4", "L5"]

# Expected output channels for each layer group
LAYER_CHANNELS: Dict[str, int] = {
    "L1": 256,
    "L2": 512,
    "L3": 1024,
    "L4": 2048,
    "L5": 2048,
}

# Expected spatial sizes (H = W) for each layer
LAYER_SPATIAL: Dict[str, int] = {
    "L1": 112,
    "L2": 56,
    "L3": 28,
    "L4": 14,
    "L5": 1,
}


class ActivationHook:
    """
    Registers forward hooks on a ResNet-50 model to capture intermediate
    activations after each of the five canonical layer groups.

    Usage
    -----
    Explicit::

        hook = ActivationHook(model)
        acts = hook.get_activations(x)   # dict[str, Tensor]
        hook.remove()

    Context-manager::

        with ActivationHook(model) as hook:
            acts = hook.get_activations(x)

    Parameters
    ----------
    model : nn.Module
        A ResNet-50 (from torchvision or compatible interface).  The model
        is expected to have attributes ``layer1``, ``layer2``, ``layer3``,
        ``layer4``, and ``avgpool``.
    pool_spatial : bool
        When *True* (default) the spatial activations of L1-L4 are globally
        average-pooled before being returned, yielding 1-D feature vectors
        of shape ``(B, C_l)``.  L5 is already ``(B, 2048, 1, 1)`` and is
        always flattened to ``(B, 2048)``.
        When *False* the full spatial tensors are returned unchanged.
    """

    def __init__(self, model: nn.Module, pool_spatial: bool = True) -> None:
        self.model = model
        self.pool_spatial = pool_spatial

        # Storage filled by hook callbacks
        self._raw: Dict[str, torch.Tensor] = {}

        # Registered hook handles (needed for cleanup)
        self._handles: List[torch.utils.hooks.RemovableHook] = []

        self._register_hooks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_hook(self, layer_id: str):
        """Return a forward-hook closure that stores the output tensor."""

        def _hook(module: nn.Module, input, output: torch.Tensor) -> None:
            # Detach to avoid retaining the computation graph
            self._raw[layer_id] = output.detach()

        return _hook

    def _register_hooks(self) -> None:
        """Attach hooks to the five named sub-modules."""
        model = self.model

        # Resolve the five target modules
        # layer{1-4} are nn.Sequential; we hook their last Bottleneck's relu.
        # avgpool is directly accessible.
        targets: Dict[str, nn.Module] = {
            "L1": model.layer1[-1].relu,   # type: ignore[index]
            "L2": model.layer2[-1].relu,   # type: ignore[index]
            "L3": model.layer3[-1].relu,   # type: ignore[index]
            "L4": model.layer4[-1].relu,   # type: ignore[index]
            "L5": model.avgpool,
        }

        for layer_id, module in targets.items():
            handle = module.register_forward_hook(self._make_hook(layer_id))
            self._handles.append(handle)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remove(self) -> None:
        """Remove all registered hooks from the model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_activations(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass through the model and return captured activations.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, 3, H, W)``.

        Returns
        -------
        dict
            Keys are ``"L1"`` ... ``"L5"``.

            * When ``pool_spatial=True`` (default):

              - L1-L4: ``(B, C_l)``  (globally average-pooled)
              - L5:    ``(B, 2048)`` (already 1x1, just squeezed)

            * When ``pool_spatial=False``:

              - L1: ``(B, 256, 112, 112)``
              - L2: ``(B, 512,  56,  56)``
              - L3: ``(B, 1024, 28,  28)``
              - L4: ``(B, 2048, 14,  14)``
              - L5: ``(B, 2048,  1,   1)``
        """
        self._raw.clear()

        with torch.no_grad():
            _ = self.model(x)

        return self._process(self._raw)

    def _process(
        self, raw: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply optional spatial pooling and return processed activations."""
        out: Dict[str, torch.Tensor] = {}

        for layer_id in LAYER_IDS:
            tensor = raw[layer_id]

            if layer_id == "L5":
                # avgpool output is (B, 2048, 1, 1) -> flatten to (B, 2048)
                out[layer_id] = tensor.flatten(1)
            elif self.pool_spatial:
                # Global average pool over (H, W) -> (B, C)
                out[layer_id] = F.adaptive_avg_pool2d(tensor, 1).flatten(1)
            else:
                out[layer_id] = tensor

        return out

    def get_spatial_activations(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience wrapper: always returns full spatial tensors (no pooling),
        regardless of the ``pool_spatial`` setting on the instance.

        Returns
        -------
        dict
            L1: ``(B, 256, 112, 112)``
            L2: ``(B, 512,  56,  56)``
            L3: ``(B, 1024, 28,  28)``
            L4: ``(B, 2048, 14,  14)``
            L5: ``(B, 2048,  1,   1)`` (not squeezed)
        """
        self._raw.clear()

        with torch.no_grad():
            _ = self.model(x)

        # Return raw tensors without pooling
        return {lid: self._raw[lid] for lid in LAYER_IDS}

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "ActivationHook":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "active" if self._handles else "removed"
        return (
            f"ActivationHook("
            f"model={type(self.model).__name__}, "
            f"pool_spatial={self.pool_spatial}, "
            f"status={status})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_activation_hook(
    model: nn.Module,
    pool_spatial: bool = True,
) -> ActivationHook:
    """
    Build and return an :class:`ActivationHook` for the given ResNet-50 model.

    Parameters
    ----------
    model : nn.Module
        ResNet-50 instance (torchvision).
    pool_spatial : bool
        Whether to spatially average-pool L1-L4 activations.

    Returns
    -------
    ActivationHook
    """
    return ActivationHook(model, pool_spatial=pool_spatial)
