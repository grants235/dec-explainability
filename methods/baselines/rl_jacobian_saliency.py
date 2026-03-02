"""
RL Saliency Baselines for MiniGrid agents.

Implements two gradient-based attribution methods:

1. Jacobian Saliency
   -----------------
   Computes the gradient of log pi(a_t | o_t) with respect to the observation:

       J_t = grad_{o_t}  log pi(a_t | o_t)

   The saliency map is the absolute value |J_t| summed (or max-pooled) over the
   channel dimension, yielding a (H, W) importance map.

   Reference: Simonyan et al., "Deep Inside Convolutional Networks" (2014).

2. Value-Difference Saliency
   --------------------------
   For each spatial cell (i, j) in the grid:
       - Replace cell (i, j) in the observation with an "empty" encoding.
       - Recompute V(o_tilde).
       - Saliency = |V(o) - V(o_tilde)|

   This is a perturbation-based approach that measures how much the value
   estimate changes when a cell is occluded.

Both methods wrap a trained SB3 PPO policy or any callable policy module.

Usage
-----
    from rl_jacobian_saliency import JacobianSaliency, ValueDiffSaliency
    from stable_baselines3 import PPO

    model = PPO.load("checkpoint/MiniGrid-DoorKey-8x8-v0")
    obs_np = ...  # (H, W, C) numpy array

    jac = JacobianSaliency()
    sal = jac.compute(model.policy, obs_tensor, action=2)
    jac.visualize(obs_np, sal, "jacobian_saliency.png")

    vd = ValueDiffSaliency()
    sal = vd.compute(model.policy, obs_tensor)
    vd.visualize(obs_np, sal, "value_diff_saliency.png")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A "policy" is any object with:
#   .predict(obs) -> (action, _)          (SB3 convention)
#   .policy.features_extractor(obs_t)     (SB3 ActorCriticPolicy)
#   .policy.mlp_extractor(features)
#   .policy.action_net(latent_pi)
#   .policy.value_net(latent_vf)
# OR a bare nn.Module with forward(obs_t) -> (logits, values).

PolicyLike = Any  # duck-typed

# Empty cell encoding for value-difference saliency perturbation
# MiniGrid: object_type=1 (empty), color=0 (red, unused), state=0
EMPTY_CELL_ENCODING = np.array([1, 0, 0], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _obs_to_tensor(
    obs: np.ndarray, device: str = "cpu"
) -> torch.Tensor:
    """
    Convert a (H, W, C) numpy observation to a (1, C, H, W) float tensor,
    normalising uint8 to [0, 1].

    Parameters
    ----------
    obs : np.ndarray  shape (H, W, C)
    device : str

    Returns
    -------
    torch.Tensor  shape (1, C, H, W)
    """
    t = torch.as_tensor(obs, dtype=torch.float32)
    if t.dtype == torch.float32 and t.max() > 1.5:
        t = t / 255.0
    # (H, W, C) -> (1, C, H, W)
    if t.ndim == 3 and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(0)
    return t.to(device)


def _get_logits_and_value_from_sb3(
    sb3_policy, obs_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run an SB3 ActorCriticPolicy and return (logits, value).

    Parameters
    ----------
    sb3_policy : stable_baselines3.common.policies.ActorCriticPolicy
    obs_tensor : torch.Tensor  shape (1, C, H, W)

    Returns
    -------
    logits : torch.Tensor  shape (1, n_actions)
    value  : torch.Tensor  shape (1, 1)
    """
    features = sb3_policy.extract_features(obs_tensor, sb3_policy.features_extractor)
    latent_pi, latent_vf = sb3_policy.mlp_extractor(features)
    logits = sb3_policy.action_net(latent_pi)
    value  = sb3_policy.value_net(latent_vf)
    return logits, value


def _is_sb3_policy(policy: Any) -> bool:
    """Heuristic: check if this looks like an SB3 PPO object or policy."""
    return (
        hasattr(policy, "action_net")
        and hasattr(policy, "value_net")
        and hasattr(policy, "features_extractor")
    )


def _policy_forward(
    policy: PolicyLike, obs_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch forward pass through either an SB3 policy or a bare nn.Module.

    Returns
    -------
    logits : (1, n_actions)
    value  : (1, 1)
    """
    if _is_sb3_policy(policy):
        return _get_logits_and_value_from_sb3(policy, obs_tensor)
    elif hasattr(policy, "policy") and _is_sb3_policy(policy.policy):
        # SB3 model object wrapping the policy
        return _get_logits_and_value_from_sb3(policy.policy, obs_tensor)
    else:
        # Assume bare nn.Module with forward(obs) -> (logits, value)
        out = policy(obs_tensor)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        # Single output: treat as logits
        return out, torch.zeros(obs_tensor.shape[0], 1, device=obs_tensor.device)


def _reduce_channel(map_chw: torch.Tensor, method: str = "sum") -> torch.Tensor:
    """
    Reduce a (C, H, W) or (1, C, H, W) tensor to (H, W) by summing or
    taking the max over channels.

    Parameters
    ----------
    map_chw : torch.Tensor  shape (..., C, H, W)
    method : "sum" | "max" | "mean"

    Returns
    -------
    torch.Tensor  shape (H, W)
    """
    if map_chw.ndim == 4:
        map_chw = map_chw.squeeze(0)  # (C, H, W)
    if method == "sum":
        return map_chw.sum(0)
    elif method == "max":
        return map_chw.abs().max(0).values
    else:
        return map_chw.mean(0)


def _normalise_saliency(sal: np.ndarray) -> np.ndarray:
    """
    Min-max normalise a saliency map to [0, 1].
    If the range is zero, returns zeros.
    """
    lo, hi = sal.min(), sal.max()
    if hi - lo < 1e-8:
        return np.zeros_like(sal)
    return (sal - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Method 1: Jacobian Saliency
# ---------------------------------------------------------------------------

class JacobianSaliency:
    """
    Jacobian / Gradient Saliency.

    Computes J_t = grad_{o_t} log pi(a_t | o_t) for a given action a_t,
    and returns |J_t| summed over channels as a (H, W) saliency map.

    Parameters
    ----------
    device : str
        "cpu" or "cuda".
    channel_reduce : str
        How to combine channels: "sum" (default), "max", or "mean".
    """

    def __init__(
        self,
        device: str = "cpu",
        channel_reduce: str = "sum",
    ) -> None:
        self.device = device
        self.channel_reduce = channel_reduce

    def compute(
        self,
        policy: PolicyLike,
        obs_tensor: torch.Tensor,
        action: int,
    ) -> np.ndarray:
        """
        Compute the Jacobian saliency map.

        Parameters
        ----------
        policy : PolicyLike
            Trained policy (SB3 model, SB3 ActorCriticPolicy, or nn.Module).
        obs_tensor : torch.Tensor
            Observation of shape (1, C, H, W) or (H, W, C) / (1, H, W, C).
            Will be converted to channel-first format internally.
        action : int
            The action whose log-probability we differentiate.

        Returns
        -------
        np.ndarray  shape (H, W)
            Absolute Jacobian saliency, normalised to [0, 1].
        """
        # Ensure correct shape and device
        if obs_tensor.ndim == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        if obs_tensor.shape[-1] in (1, 3) and obs_tensor.ndim == 4:
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        obs_tensor = obs_tensor.float().to(self.device)
        if obs_tensor.max() > 1.5:
            obs_tensor = obs_tensor / 255.0

        obs_tensor = obs_tensor.clone().requires_grad_(True)

        # Forward pass
        logits, _ = _policy_forward(policy, obs_tensor)           # (1, n_actions)
        log_probs = F.log_softmax(logits, dim=-1)                 # (1, n_actions)
        log_prob_a = log_probs[0, action]                         # scalar

        # Backward pass
        log_prob_a.backward()

        with torch.no_grad():
            grad = obs_tensor.grad.abs()                          # (1, C, H, W)
            sal_chw = grad.squeeze(0)                             # (C, H, W)
            sal_hw = _reduce_channel(sal_chw, self.channel_reduce).cpu().numpy()

        return _normalise_saliency(sal_hw)

    def compute_batch(
        self,
        policy: PolicyLike,
        obs_list: list,
        action_list: list,
    ) -> list:
        """
        Compute Jacobian saliency for a list of (obs, action) pairs.

        Parameters
        ----------
        policy : PolicyLike
        obs_list : list of np.ndarray  shape (H, W, C) each
        action_list : list of int

        Returns
        -------
        List of np.ndarray  shape (H, W) each.
        """
        results = []
        for obs_np, action in zip(obs_list, action_list):
            obs_t = _obs_to_tensor(obs_np, self.device)
            sal = self.compute(policy, obs_t, action)
            results.append(sal)
        return results

    def visualize(
        self,
        obs: np.ndarray,
        saliency: np.ndarray,
        save_path: str,
        title: str = "Jacobian Saliency",
        alpha: float = 0.55,
        cmap: str = "hot",
    ) -> None:
        """
        Overlay the saliency map on the observation image and save.

        Parameters
        ----------
        obs : np.ndarray  shape (H, W, C) or (H, W)
            The original observation (uint8 or float).
        saliency : np.ndarray  shape (H, W)
            Saliency map, values in [0, 1].
        save_path : str
            Output file path (PNG).
        title : str
        alpha : float
            Transparency of the saliency overlay.
        cmap : str
            Colormap for the saliency heatmap.
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Normalise observation for display
        if obs.dtype == np.uint8:
            obs_disp = obs.astype(np.float32) / 255.0
        else:
            obs_disp = obs.astype(np.float32)
        if obs_disp.ndim == 3 and obs_disp.shape[2] == 3:
            obs_rgb = obs_disp
        elif obs_disp.ndim == 3:
            obs_rgb = obs_disp[:, :, 0]
        else:
            obs_rgb = obs_disp

        # Panel 1: original observation
        axes[0].imshow(obs_rgb if obs_rgb.ndim == 3 else obs_rgb, cmap="gray")
        axes[0].set_title("Observation", fontsize=10)
        axes[0].axis("off")

        # Panel 2: saliency heatmap
        im = axes[1].imshow(saliency, cmap=cmap, vmin=0, vmax=1)
        axes[1].set_title(title, fontsize=10)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: overlay
        if obs_rgb.ndim == 2:
            base = np.stack([obs_rgb] * 3, axis=-1)
        else:
            base = obs_rgb
        # Upscale saliency to match obs size if needed
        if saliency.shape != obs_rgb.shape[:2]:
            from PIL import Image
            sal_pil = Image.fromarray((saliency * 255).astype(np.uint8))
            sal_pil = sal_pil.resize(
                (obs_rgb.shape[1], obs_rgb.shape[0]), Image.BILINEAR
            )
            sal_up = np.array(sal_pil).astype(np.float32) / 255.0
        else:
            sal_up = saliency

        colormap = plt.get_cmap(cmap)
        sal_rgba = colormap(sal_up)[:, :, :3]  # (H, W, 3)
        overlay = (1 - alpha) * base + alpha * sal_rgba
        overlay = np.clip(overlay, 0, 1)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=10)
        axes[2].axis("off")

        plt.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[JacobianSaliency.visualize] Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Method 2: Value-Difference Saliency
# ---------------------------------------------------------------------------

class ValueDiffSaliency:
    """
    Value-Difference (Perturbation) Saliency.

    For each cell (i, j) in the observation grid, replaces the cell's
    encoding with EMPTY_CELL_ENCODING and measures the absolute change in
    the value estimate:

        saliency(i, j) = |V(obs) - V(obs_perturbed_{i,j})|

    This operates on the raw channel-last observation (H, W, C) and
    perturbs one cell at a time.

    Parameters
    ----------
    device : str
        "cpu" or "cuda".
    empty_encoding : np.ndarray  shape (C,)
        The encoding to use for occluded cells.  Defaults to
        EMPTY_CELL_ENCODING = [1, 0, 0] (MiniGrid empty cell).
    batch_size : int
        Number of perturbed observations to evaluate in parallel (GPU batch).
        Set to 1 for sequential evaluation on CPU.
    """

    def __init__(
        self,
        device: str = "cpu",
        empty_encoding: Optional[np.ndarray] = None,
        batch_size: int = 32,
    ) -> None:
        self.device = device
        self.empty_encoding = (
            empty_encoding if empty_encoding is not None
            else EMPTY_CELL_ENCODING.copy()
        )
        self.batch_size = batch_size

    def compute(
        self,
        policy: PolicyLike,
        obs_tensor: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute the value-difference saliency map.

        Parameters
        ----------
        policy : PolicyLike
        obs_tensor : torch.Tensor
            Shape (1, C, H, W) or (H, W, C) / (1, H, W, C).

        Returns
        -------
        np.ndarray  shape (H, W)
            Value-difference saliency, normalised to [0, 1].
        """
        # Convert to channel-first numpy for perturbation
        if obs_tensor.ndim == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        if obs_tensor.shape[-1] in (1, 3) and obs_tensor.ndim == 4:
            # (1, H, W, C)
            obs_np_hwc = obs_tensor.squeeze(0).cpu().numpy()          # (H, W, C)
            obs_np_chw = obs_np_hwc.transpose(2, 0, 1)                # (C, H, W)
        else:
            obs_np_chw = obs_tensor.squeeze(0).cpu().numpy()          # (C, H, W)
            obs_np_hwc = obs_np_chw.transpose(1, 2, 0)                # (H, W, C)

        C, H, W = obs_np_chw.shape

        # Baseline value
        obs_base_t = torch.as_tensor(obs_np_chw, dtype=torch.float32).unsqueeze(0).to(self.device)
        if obs_base_t.max() > 1.5:
            obs_base_t = obs_base_t / 255.0
        with torch.no_grad():
            _, value_base = _policy_forward(policy, obs_base_t)
            v_base = value_base.item()

        saliency = np.zeros((H, W), dtype=np.float32)

        # Build all perturbed observations at once for batched evaluation
        n_cells = H * W
        perturbed_batch: list = []
        cell_indices: list = []

        empty_enc = self.empty_encoding  # (C,) or shorter

        for i in range(H):
            for j in range(W):
                obs_pert = obs_np_chw.copy().astype(np.float32)
                for c in range(min(C, len(empty_enc))):
                    obs_pert[c, i, j] = float(empty_enc[c])
                perturbed_batch.append(obs_pert)
                cell_indices.append((i, j))

        # Evaluate in mini-batches
        n_batches = (n_cells + self.batch_size - 1) // self.batch_size
        for b in range(n_batches):
            start = b * self.batch_size
            end   = min(start + self.batch_size, n_cells)
            batch_np = np.stack(perturbed_batch[start:end], axis=0)  # (B, C, H, W)
            batch_t  = torch.as_tensor(batch_np, dtype=torch.float32).to(self.device)
            if batch_t.max() > 1.5:
                batch_t = batch_t / 255.0

            with torch.no_grad():
                _, values_t = _policy_forward(policy, batch_t)        # (B, 1)
            values_np = values_t.cpu().numpy().flatten()               # (B,)

            for rel_idx, (ii, jj) in enumerate(cell_indices[start:end]):
                saliency[ii, jj] = abs(v_base - float(values_np[rel_idx]))

        return _normalise_saliency(saliency)

    def compute_from_obs(
        self,
        policy: PolicyLike,
        obs: np.ndarray,
    ) -> np.ndarray:
        """
        Convenience wrapper: accepts a (H, W, C) numpy observation directly.

        Parameters
        ----------
        policy : PolicyLike
        obs : np.ndarray  shape (H, W, C)

        Returns
        -------
        np.ndarray  shape (H, W)
        """
        obs_t = _obs_to_tensor(obs, self.device)
        return self.compute(policy, obs_t)

    def visualize(
        self,
        obs: np.ndarray,
        saliency: np.ndarray,
        save_path: str,
        title: str = "Value-Difference Saliency",
        alpha: float = 0.55,
        cmap: str = "plasma",
    ) -> None:
        """
        Overlay the value-difference saliency on the observation and save.

        Parameters
        ----------
        obs : np.ndarray  shape (H, W, C) or (H, W)
        saliency : np.ndarray  shape (H, W)
            Saliency map in [0, 1].
        save_path : str
        title : str
        alpha : float
        cmap : str
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        if obs.dtype == np.uint8:
            obs_disp = obs.astype(np.float32) / 255.0
        else:
            obs_disp = obs.astype(np.float32)

        if obs_disp.ndim == 3 and obs_disp.shape[2] == 3:
            obs_rgb = obs_disp
        elif obs_disp.ndim == 3:
            obs_rgb = obs_disp[:, :, 0]
        else:
            obs_rgb = obs_disp

        # Panel 1: observation
        axes[0].imshow(obs_rgb if obs_rgb.ndim == 3 else obs_rgb, cmap="gray")
        axes[0].set_title("Observation", fontsize=10)
        axes[0].axis("off")

        # Panel 2: saliency heatmap
        im = axes[1].imshow(saliency, cmap=cmap, vmin=0, vmax=1)
        axes[1].set_title(title, fontsize=10)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: overlay
        if obs_rgb.ndim == 2:
            base = np.stack([obs_rgb] * 3, axis=-1)
        else:
            base = obs_rgb

        if saliency.shape != obs_rgb.shape[:2]:
            from PIL import Image
            sal_pil = Image.fromarray((saliency * 255).astype(np.uint8))
            sal_pil = sal_pil.resize(
                (obs_rgb.shape[1], obs_rgb.shape[0]), Image.BILINEAR
            )
            sal_up = np.array(sal_pil).astype(np.float32) / 255.0
        else:
            sal_up = saliency

        colormap = plt.get_cmap(cmap)
        sal_rgba = colormap(sal_up)[:, :, :3]
        overlay = (1 - alpha) * base + alpha * sal_rgba
        overlay = np.clip(overlay, 0, 1)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=10)
        axes[2].axis("off")

        plt.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[ValueDiffSaliency.visualize] Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Combined comparison visualisation
# ---------------------------------------------------------------------------

def visualize_comparison(
    obs: np.ndarray,
    jacobian_saliency: np.ndarray,
    value_diff_saliency: np.ndarray,
    save_path: str,
    title: str = "Saliency Baseline Comparison",
) -> None:
    """
    Side-by-side comparison: observation | Jacobian | Value-Diff | overlays.

    Parameters
    ----------
    obs : np.ndarray  shape (H, W, C)
    jacobian_saliency : np.ndarray  shape (H, W)
    value_diff_saliency : np.ndarray  shape (H, W)
    save_path : str
    title : str
    """
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    if obs.dtype == np.uint8:
        obs_disp = obs.astype(np.float32) / 255.0
    else:
        obs_disp = obs.astype(np.float32)
    obs_rgb = obs_disp if obs_disp.ndim == 3 and obs_disp.shape[2] == 3 else obs_disp

    def _overlay(base, sal, cmap="hot", alpha=0.55):
        if base.ndim == 2:
            base = np.stack([base] * 3, axis=-1)
        cm = plt.get_cmap(cmap)
        rgba = cm(sal)[:, :, :3]
        return np.clip((1 - alpha) * base + alpha * rgba, 0, 1)

    labels = [
        "Observation",
        "Jacobian Saliency",
        "Jacobian Overlay",
        "Value-Diff Saliency",
        "Value-Diff Overlay",
    ]
    images = [
        obs_rgb,
        jacobian_saliency,
        _overlay(obs_rgb, jacobian_saliency, "hot"),
        value_diff_saliency,
        _overlay(obs_rgb, value_diff_saliency, "plasma"),
    ]
    cmaps = ["gray" if obs_rgb.ndim == 2 else None, "hot", None, "plasma", None]

    for ax, img, lbl, cm in zip(axes, images, labels, cmaps):
        if cm is not None:
            ax.imshow(img, cmap=cm, vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(lbl, fontsize=9)
        ax.axis("off")

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize_comparison] Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Trajectory-level saliency computation helpers
# ---------------------------------------------------------------------------

def compute_trajectory_saliency(
    policy: PolicyLike,
    trajectory: Dict[str, Any],
    method: str = "jacobian",
    device: str = "cpu",
    **kwargs: Any,
) -> list:
    """
    Compute saliency maps for every timestep in a trajectory.

    Parameters
    ----------
    policy : PolicyLike
    trajectory : dict  with keys obs_seq, action_seq
    method : "jacobian" | "value_diff"
    device : str
    **kwargs : passed to the saliency method constructor.

    Returns
    -------
    List[np.ndarray]  shape (H, W) per timestep.
    """
    obs_seq: list = trajectory["obs_seq"]
    action_seq: list = trajectory.get("action_seq", [0] * len(obs_seq))

    if method == "jacobian":
        saliency_fn = JacobianSaliency(device=device, **kwargs)
    elif method == "value_diff":
        saliency_fn = ValueDiffSaliency(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'jacobian' or 'value_diff'.")

    results = []
    for t, obs_np in enumerate(obs_seq):
        obs_t = _obs_to_tensor(obs_np, device)
        if method == "jacobian":
            action = action_seq[t] if t < len(action_seq) else 0
            sal = saliency_fn.compute(policy, obs_t, action)
        else:
            sal = saliency_fn.compute(policy, obs_t)
        results.append(sal)

    return results


def visualize_trajectory_saliency(
    obs_seq: list,
    saliency_seq: list,
    save_path: str,
    method_name: str = "Jacobian",
    max_frames: int = 20,
    cmap: str = "hot",
) -> None:
    """
    Grid visualisation of (observation, saliency overlay) pairs for selected
    timesteps in a trajectory.

    Parameters
    ----------
    obs_seq : list of np.ndarray  (H, W, C)
    saliency_seq : list of np.ndarray  (H, W)
    save_path : str
    method_name : str
    max_frames : int  – show at most this many frames
    cmap : str
    """
    T = min(len(obs_seq), len(saliency_seq), max_frames)
    step = max(1, len(obs_seq) // max_frames)
    indices = list(range(0, len(obs_seq), step))[:max_frames]
    n = len(indices)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 4))
    if n == 1:
        axes = axes.reshape(2, 1)

    colormap = plt.get_cmap(cmap)

    for col, t in enumerate(indices):
        obs_np = obs_seq[t]
        sal = saliency_seq[t]

        if obs_np.dtype == np.uint8:
            obs_disp = obs_np.astype(np.float32) / 255.0
        else:
            obs_disp = obs_np.astype(np.float32)

        obs_rgb = obs_disp if (obs_disp.ndim == 3 and obs_disp.shape[2] == 3) else obs_disp

        # Row 0: observation
        axes[0, col].imshow(obs_rgb if obs_rgb.ndim == 3 else obs_rgb, cmap="gray")
        axes[0, col].set_title(f"t={t}", fontsize=7)
        axes[0, col].axis("off")

        # Row 1: saliency overlay
        if obs_rgb.ndim == 2:
            base = np.stack([obs_rgb] * 3, axis=-1)
        else:
            base = obs_rgb
        sal_rgba = colormap(sal)[:, :, :3]
        overlay = np.clip(0.45 * base + 0.55 * sal_rgba, 0, 1)
        axes[1, col].imshow(overlay)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Obs", fontsize=8)
    axes[1, 0].set_ylabel(f"{method_name} Sal.", fontsize=8)

    plt.suptitle(f"{method_name} Saliency — Trajectory", fontsize=11, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize_trajectory_saliency] Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import gymnasium as gym
    from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

    print("=== RL Saliency Baseline Demo ===")
    env_name = "MiniGrid-DoorKey-8x8-v0"
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    obs, _ = env.reset()
    obs_h, obs_w, obs_c = env.observation_space.shape
    n_actions = env.action_space.n

    print(f"Obs shape: {obs.shape}, n_actions: {n_actions}")

    # Build a minimal dummy policy (random weights)
    class DummyPolicy(nn.Module):
        def __init__(self, n_in_ch, h, w, n_act):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(n_in_ch, 16, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, n_in_ch, h, w)
                flat = self.cnn(dummy).shape[1]
            self.fc = nn.Linear(flat, 256)
            self.policy_head = nn.Linear(256, n_act)
            self.value_head  = nn.Linear(256, 1)

        def forward(self, x):
            if x.ndim == 4 and x.shape[-1] in (1, 3):
                x = x.permute(0, 3, 1, 2)
            x = x.float() / 255.0 if x.max() > 1.5 else x.float()
            h = torch.relu(self.fc(self.cnn(x)))
            return self.policy_head(h), self.value_head(h)

    policy = DummyPolicy(obs_c, obs_h, obs_w, n_actions)

    # Jacobian saliency
    jac = JacobianSaliency(device="cpu")
    obs_t = _obs_to_tensor(obs, "cpu")
    sal_jac = jac.compute(policy, obs_t, action=2)  # FORWARD
    print(f"Jacobian saliency shape: {sal_jac.shape}, range: [{sal_jac.min():.3f}, {sal_jac.max():.3f}]")
    jac.visualize(obs, sal_jac, "/tmp/jacobian_demo.png")

    # Value-difference saliency
    vd = ValueDiffSaliency(device="cpu", batch_size=16)
    sal_vd = vd.compute_from_obs(policy, obs)
    print(f"Value-diff saliency shape: {sal_vd.shape}, range: [{sal_vd.min():.3f}, {sal_vd.max():.3f}]")
    vd.visualize(obs, sal_vd, "/tmp/value_diff_demo.png")

    # Comparison
    visualize_comparison(obs, sal_jac, sal_vd, "/tmp/saliency_comparison_demo.png")

    env.close()
    print("Demo complete.")
