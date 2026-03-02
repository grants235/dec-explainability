"""
Evaluation metrics for teleological explainability.

Covers both image (CUB-200) and RL (MiniGrid) settings:

Image metrics
-------------
- compute_pbpa                  : Part-Based Purposive Alignment
- compute_deletion_auc          : Deletion AUC
- compute_insertion_auc         : Insertion AUC
- compute_purposive_specificity : Purposive Specificity (SSIM-based)
- compute_means_end_coherence   : Means-End Coherence (monotonicity +
                                  consumption consistency)

RL metrics
----------
- compute_segmentation_f1        : Sub-goal segmentation boundary F1
- compute_counterfactual_validity : Counterfactual validity + trajectory
                                   action divergence
- compute_gn_entropy_correlation  : GN-quartile entropy reduction
- compute_human_proxy_predictability : MLP-based human-proxy predictability
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, wilcoxon
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# CUB-200 Part / Attribute mapping helpers
# ---------------------------------------------------------------------------

# CUB-200-2011 part names (15 canonical parts)
CUB_PART_NAMES: List[str] = [
    "back",
    "beak",
    "belly",
    "breast",
    "crown",
    "forehead",
    "left_eye",
    "left_leg",
    "left_wing",
    "nape",
    "right_eye",
    "right_leg",
    "right_wing",
    "tail",
    "throat",
]

# Mapping from CUB part name -> canonical attribute group name
PART_TO_ATTR: Dict[str, str] = {
    "beak": "beak",
    "crown": "crown",
    "forehead": "crown",
    "nape": "crown",
    "left_eye": "left_eye",
    "right_eye": "right_eye",
    "throat": "throat",
    "breast": "breast",
    "belly": "belly",
    "back": "back",
    "left_wing": "left_wing",
    "right_wing": "right_wing",
    "tail": "tail",
    "left_leg": "left_leg",
    "right_leg": "right_leg",
}


# ---------------------------------------------------------------------------
# Image metric 1: PBPA (Part-Based Purposive Alignment)
# ---------------------------------------------------------------------------

def compute_pbpa(
    saliency_map: np.ndarray,
    part_keypoints: Dict[str, Optional[Tuple[int, int]]],
    diagnostic_parts: List[str],
    image_size: int = 448,
    radius: int = 30,
    percentile: float = 90.0,
) -> float:
    """
    Part-Based Purposive Alignment (PBPA).

    Measures the fraction of high-saliency pixels that fall within
    circular regions (radius *r* px) around visible keypoints of
    the diagnostic body parts.

    Parameters
    ----------
    saliency_map : np.ndarray
        Shape ``(image_size, image_size)``, non-negative float.
    part_keypoints : dict
        Mapping from part name (CUB conventions) to ``(row, col)``
        pixel coordinates, or ``None`` if the part is not visible.
    diagnostic_parts : list of str
        Part names that are considered diagnostic for the confusion
        pair being explained.
    image_size : int
        Expected spatial size of the saliency map (default 448).
    radius : int
        Radius in pixels of the circular region around each keypoint
        (default 30).
    percentile : float
        Percentile threshold for "high-saliency" pixels (default 90).

    Returns
    -------
    float
        PBPA score in [0, 1].  Returns 0.0 if no diagnostic keypoint is
        visible or if there are no high-saliency pixels.
    """
    if saliency_map.shape != (image_size, image_size):
        saliency_map = _resize_map(saliency_map, image_size)

    # Threshold: pixels above 90th percentile
    threshold = np.percentile(saliency_map, percentile)
    high_mask = saliency_map > threshold

    n_high = high_mask.sum()
    if n_high == 0:
        return 0.0

    # Build circular masks for all visible diagnostic keypoints
    ys, xs = np.mgrid[0:image_size, 0:image_size]
    part_mask = np.zeros((image_size, image_size), dtype=bool)

    any_visible = False
    for part in diagnostic_parts:
        kp = part_keypoints.get(part)
        if kp is None:
            continue
        row, col = int(kp[0]), int(kp[1])
        circle = (ys - row) ** 2 + (xs - col) ** 2 <= radius ** 2
        part_mask |= circle
        any_visible = True

    if not any_visible:
        return 0.0

    # Fraction of high-saliency pixels inside diagnostic regions
    pbpa = float((high_mask & part_mask).sum()) / float(n_high)
    return float(np.clip(pbpa, 0.0, 1.0))


def compute_diagnostic_parts(
    class_i: int,
    class_j: int,
    class_attribute_matrix: np.ndarray,
    part_to_attr_indices: Dict[str, List[int]],
    eta: Optional[float] = None,
    target_n_diagnostic: float = 4.0,
) -> Tuple[List[str], float]:
    """
    Determine which parts are diagnostic for the (i, j) confusion pair.

    For each part *p*, compute:

        d_p(i, j) = mean over attrs_p of |E[attr | Y=i] - E[attr | Y=j]|

    A part is diagnostic if ``d_p > eta``.  If ``eta`` is None it is
    chosen adaptively so that on average ``target_n_diagnostic`` parts
    are diagnostic per pair.

    Parameters
    ----------
    class_i, class_j : int
        Class indices (0-based).
    class_attribute_matrix : np.ndarray
        Shape ``(n_classes, n_attributes)``.  Each entry is the
        mean attribute activation for that class.
    part_to_attr_indices : dict
        Mapping from part name -> list of attribute column indices
        that correspond to that part.
    eta : float or None
        Threshold.  Computed adaptively when None.
    target_n_diagnostic : float
        Desired average number of diagnostic parts (default 4.0).

    Returns
    -------
    diagnostic_parts : list of str
    eta : float
    """
    scores: Dict[str, float] = {}
    for part, attr_idxs in part_to_attr_indices.items():
        if not attr_idxs:
            continue
        diffs = np.abs(
            class_attribute_matrix[class_i, attr_idxs]
            - class_attribute_matrix[class_j, attr_idxs]
        )
        scores[part] = float(diffs.mean())

    if not scores:
        return [], 0.0

    score_vals = np.array(list(scores.values()))

    if eta is None:
        # Adaptive threshold: choose eta as the (1 - target_n/total) quantile
        frac = 1.0 - min(target_n_diagnostic / max(len(scores), 1), 1.0)
        frac = float(np.clip(frac, 0.0, 1.0))
        eta = float(np.quantile(score_vals, frac))

    diagnostic = [p for p, s in scores.items() if s > eta]
    return diagnostic, float(eta)


# ---------------------------------------------------------------------------
# Image metric 2: Deletion AUC
# ---------------------------------------------------------------------------

def compute_deletion_auc(
    model: nn.Module,
    x: torch.Tensor,
    saliency_map: np.ndarray,
    true_class: int,
    device: torch.device,
    n_steps: int = 21,
    mask_value: Optional[torch.Tensor] = None,
) -> float:
    """
    Deletion AUC.

    Progressively masks the *most salient* pixels (replacing them with
    ``mask_value``) and records the model confidence in ``true_class``
    at each step.  AUC is computed via the trapezoid rule.

    A *lower* deletion AUC indicates a better explanation (the classifier
    drops quickly when the important region is removed).

    Parameters
    ----------
    model : nn.Module
        Trained classifier (eval mode).
    x : torch.Tensor
        Single image tensor ``(1, 3, H, W)``.
    saliency_map : np.ndarray
        Shape ``(H, W)``.  Higher values = more salient.
    true_class : int
        Target class for measuring confidence.
    device : torch.device
    n_steps : int
        Number of masking steps (including 0 % and 100 %); default 21
        gives steps at k = 0, 5, 10, …, 100 %.
    mask_value : torch.Tensor or None
        Value to fill masked pixels.  Shape must broadcast to
        ``(3, H, W)`` or ``(1, 3, H, W)``.  If None a dataset-mean
        ImageNet tensor is used.

    Returns
    -------
    float
        Deletion AUC (trapezoidal).
    """
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)

    if mask_value is None:
        mask_value = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    else:
        mask_value = mask_value.to(device)

    H, W = x.shape[2], x.shape[3]
    n_pixels = H * W

    # Flatten saliency map and sort descending
    flat_sal = saliency_map.flatten()
    sorted_idx = np.argsort(flat_sal)[::-1]  # most salient first

    percentages = np.linspace(0, 1, n_steps)
    confidences = []

    with torch.no_grad():
        for pct in percentages:
            n_mask = int(pct * n_pixels)
            x_masked = x.clone()

            if n_mask > 0:
                mask_flat = np.zeros(n_pixels, dtype=bool)
                mask_flat[sorted_idx[:n_mask]] = True
                mask_2d = torch.from_numpy(mask_flat.reshape(H, W)).to(device)  # (H, W)
                mask_3d = mask_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                x_masked = torch.where(mask_3d, mask_value.expand_as(x), x_masked)

            logits = model(x_masked)
            probs = torch.softmax(logits, dim=-1)
            conf = probs[0, true_class].item()
            confidences.append(conf)

    auc = float(np.trapz(confidences, percentages))
    return auc


# ---------------------------------------------------------------------------
# Image metric 3: Insertion AUC
# ---------------------------------------------------------------------------

def compute_insertion_auc(
    model: nn.Module,
    x: torch.Tensor,
    saliency_map: np.ndarray,
    true_class: int,
    device: torch.device,
    n_steps: int = 21,
) -> float:
    """
    Insertion AUC.

    Starts from a fully-masked (dataset-mean) image and progressively
    *reveals* the most salient pixels.  Records model confidence in
    ``true_class`` at each step.

    A *higher* insertion AUC indicates a better explanation.

    Parameters
    ----------
    model : nn.Module
    x : torch.Tensor
        Shape ``(1, 3, H, W)``.
    saliency_map : np.ndarray
        Shape ``(H, W)``.
    true_class : int
    device : torch.device
    n_steps : int
        Default 21.

    Returns
    -------
    float
        Insertion AUC (trapezoidal).
    """
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)

    # Mean image (ImageNet mean)
    mean_val = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    mean_image = mean_val.expand_as(x)

    H, W = x.shape[2], x.shape[3]
    n_pixels = H * W

    flat_sal = saliency_map.flatten()
    sorted_idx = np.argsort(flat_sal)[::-1]  # most salient first

    percentages = np.linspace(0, 1, n_steps)
    confidences = []

    with torch.no_grad():
        for pct in percentages:
            n_reveal = int(pct * n_pixels)
            x_revealed = mean_image.clone()

            if n_reveal > 0:
                reveal_flat = np.zeros(n_pixels, dtype=bool)
                reveal_flat[sorted_idx[:n_reveal]] = True
                reveal_2d = torch.from_numpy(reveal_flat.reshape(H, W)).to(device)
                reveal_3d = reveal_2d.unsqueeze(0).unsqueeze(0)
                x_revealed = torch.where(reveal_3d, x, x_revealed)

            logits = model(x_revealed)
            probs = torch.softmax(logits, dim=-1)
            conf = probs[0, true_class].item()
            confidences.append(conf)

    auc = float(np.trapz(confidences, percentages))
    return auc


# ---------------------------------------------------------------------------
# Image metric 4: Purposive Specificity
# ---------------------------------------------------------------------------

def compute_purposive_specificity(
    per_competitor_maps: List[np.ndarray],
) -> float:
    """
    Purposive Specificity (PS).

    Measures how distinct the saliency maps are across different
    competitor classes.  A high score means the explanation changes
    meaningfully when the confusion target changes.

        PS(x) = 1 - (1 / C(k,2)) * sum_{j < l} SSIM(|S_j|, |S_l|)

    where S_j is the saliency map for competitor j.

    Parameters
    ----------
    per_competitor_maps : list of np.ndarray
        Each element is a ``(H, W)`` float array (absolute-valued
        saliency).  k must be >= 2.

    Returns
    -------
    float
        PS in [0, 1].  Returns 0.0 if fewer than 2 maps are provided.
    """
    k = len(per_competitor_maps)
    if k < 2:
        return 0.0

    ssim_sum = 0.0
    n_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            m_i = np.abs(per_competitor_maps[i]).astype(np.float64)
            m_j = np.abs(per_competitor_maps[j]).astype(np.float64)

            # Normalise each map to [0,1] for SSIM
            def _norm(m: np.ndarray) -> np.ndarray:
                r = m.max() - m.min()
                return (m - m.min()) / (r + 1e-8)

            m_i = _norm(m_i)
            m_j = _norm(m_j)

            val = ssim(m_i, m_j, data_range=1.0)
            ssim_sum += val
            n_pairs += 1

    mean_ssim = ssim_sum / n_pairs
    ps = 1.0 - mean_ssim
    return float(np.clip(ps, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Image metric 5: Means-End Coherence
# ---------------------------------------------------------------------------

def compute_means_end_coherence(
    T_layers: List[np.ndarray],
    Delta_layers: List[np.ndarray],
    kappa_layers: List[np.ndarray],
    tolerance: float = 0.02,
) -> Dict[str, float]:
    """
    Means-End Coherence.

    Combines two sub-measures:

    1. **Monotonicity** – fraction of consecutive-layer pairs where
       ``D_{l+1} >= D_l - tolerance`` (i.e. discriminability does not
       decrease sharply).

    2. **Consumption Consistency** – Pearson correlation between the
       feature-consumption rate ``kappa_l`` and the discriminability
       gain ``Delta_{l-1}`` across layers.  Positive correlation means
       layers that consume more features produce larger discriminability
       gains.

    Parameters
    ----------
    T_layers : list of np.ndarray
        Discriminability matrices D_l, each of shape ``(n_classes,
        n_classes)`` or a 1-D array of per-pair scores for layer l.
    Delta_layers : list of np.ndarray
        Discriminability *gain* matrices Delta_l = D_l - D_{l-1}.
        Must have the same length as ``T_layers``.
    kappa_layers : list of np.ndarray
        Feature consumption rates kappa_l for each layer.
        Must have the same length as ``T_layers``.
    tolerance : float
        Slack for the monotonicity check (default 0.02).

    Returns
    -------
    dict with keys:
        'monotonicity'           : float in [0, 1]
        'consumption_consistency': float in [-1, 1]
        'coherence'              : float in [0, 1]  (geometric mean)
    """
    L = len(T_layers)
    if L < 2:
        return {"monotonicity": 1.0, "consumption_consistency": 0.0, "coherence": 0.0}

    # ---- Monotonicity -------------------------------------------------------
    monotone_count = 0
    total_pairs = 0
    for l in range(L - 1):
        d_l = np.asarray(T_layers[l]).flatten()
        d_lp1 = np.asarray(T_layers[l + 1]).flatten()
        pair_mono = (d_lp1 >= d_l - tolerance).mean()
        monotone_count += float(pair_mono)
        total_pairs += 1

    monotonicity = monotone_count / total_pairs if total_pairs > 0 else 1.0

    # ---- Consumption Consistency --------------------------------------------
    # Align: Delta_{l} pairs with kappa_{l} for l = 1 .. L-1
    # Delta_layers[0] corresponds to Delta_1 = D_1 - D_0 (no kappa_0 pairing)
    # We correlate kappa_l with Delta_{l} for l = 1 .. L-1
    delta_vals: List[float] = []
    kappa_vals: List[float] = []

    for l in range(1, L):
        delta_l = float(np.mean(np.abs(np.asarray(Delta_layers[l]).flatten())))
        kappa_l = float(np.mean(np.abs(np.asarray(kappa_layers[l]).flatten())))
        delta_vals.append(delta_l)
        kappa_vals.append(kappa_l)

    if len(delta_vals) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = pearsonr(kappa_vals, delta_vals)
        if np.isnan(corr):
            corr = 0.0
        consumption_consistency = float(corr)
    else:
        consumption_consistency = 0.0

    # ---- Overall coherence (geometric mean of monotonicity and scaled CC) ---
    cc_scaled = (consumption_consistency + 1.0) / 2.0  # map [-1,1] -> [0,1]
    coherence = float(np.sqrt(monotonicity * cc_scaled))

    return {
        "monotonicity": float(np.clip(monotonicity, 0.0, 1.0)),
        "consumption_consistency": float(consumption_consistency),
        "coherence": float(np.clip(coherence, 0.0, 1.0)),
    }


# ---------------------------------------------------------------------------
# RL metric 1: Sub-Goal Segmentation F1
# ---------------------------------------------------------------------------

def compute_segmentation_f1(
    predicted_segments: List[Tuple[int, int]],
    gt_segments: List[Tuple[int, int]],
    tolerance: int = 2,
) -> Dict[str, float]:
    """
    Sub-goal segmentation boundary F1 with temporal tolerance.

    Parameters
    ----------
    predicted_segments : list of (start, end) tuples
        Predicted segment boundaries (inclusive, 0-based timesteps).
    gt_segments : list of (start, end) tuples
        Ground-truth segments from BFS (inclusive, 0-based timesteps).
    tolerance : int
        Timestep tolerance for matching boundary points (default 2).

    Returns
    -------
    dict with keys:
        'precision' : float
        'recall'    : float
        'f1'        : float
    """
    def _boundaries(segs: List[Tuple[int, int]]) -> List[int]:
        """Extract all unique boundary timesteps (start and end)."""
        bounds = set()
        for s, e in segs:
            bounds.add(s)
            bounds.add(e)
        return sorted(bounds)

    pred_bounds = _boundaries(predicted_segments)
    gt_bounds = _boundaries(gt_segments)

    # Match predicted boundaries to ground truth within tolerance
    matched_pred = set()
    matched_gt = set()
    for i, pb in enumerate(pred_bounds):
        for j, gb in enumerate(gt_bounds):
            if j not in matched_gt and abs(pb - gb) <= tolerance:
                matched_pred.add(i)
                matched_gt.add(j)
                break

    tp = len(matched_pred)
    precision = tp / len(pred_bounds) if pred_bounds else 0.0
    recall = tp / len(gt_bounds) if gt_bounds else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


# ---------------------------------------------------------------------------
# RL metric 2: Counterfactual Validity
# ---------------------------------------------------------------------------

def compute_counterfactual_validity(
    gc_policy: Any,
    trajectories: List[Dict[str, Any]],
    n_samples: int = 100,
    horizon: int = 20,
) -> Dict[str, float]:
    """
    Counterfactual Validity.

    For each sampled trajectory, conditions the goal-conditioned policy
    on an *alternative* goal (randomly chosen from the subgoal set) and
    checks whether the first predicted action diverges from the baseline.

    Also computes trajectory-level action divergence over ``horizon``
    steps.

    Parameters
    ----------
    gc_policy : Any
        Goal-conditioned policy object with a method::

            predict(obs, goal) -> action (int)

        or::

            get_action_distribution(obs, goal) -> np.ndarray (probabilities)

    trajectories : list of dict
        Each dict should have at minimum:
        - 'obs'         : np.ndarray (T, obs_dim)
        - 'actions'     : np.ndarray (T,) int
        - 'subgoals'    : list of goal descriptors, length T
        - 'alt_goals'   : list of alternative goal descriptors, length T

    n_samples : int
        Maximum number of trajectory steps to evaluate (default 100).
    horizon : int
        Number of steps for trajectory-level divergence (default 20).

    Returns
    -------
    dict with keys:
        'first_action_agreement' : float   fraction of steps where CF first
                                           action equals baseline first action
        'mean_trajectory_divergence' : float   mean action divergence rate
                                               over *horizon* steps
    """
    first_action_agreements = []
    traj_divergences = []

    rng = np.random.default_rng(42)
    sample_count = 0

    for traj in trajectories:
        obs = traj["obs"]  # (T, obs_dim)
        actions = traj["actions"]  # (T,)
        subgoals = traj["subgoals"]
        alt_goals = traj.get("alt_goals", subgoals)  # fallback

        T = len(actions)
        step_indices = rng.choice(T, size=min(n_samples, T), replace=False)

        for t in step_indices:
            if sample_count >= n_samples:
                break
            o = obs[t]
            baseline_goal = subgoals[t]
            alt_goal = alt_goals[t]

            # First-action agreement
            if hasattr(gc_policy, "predict"):
                baseline_action = gc_policy.predict(o, baseline_goal)
                alt_action = gc_policy.predict(o, alt_goal)
                first_action_agreements.append(int(baseline_action == alt_action))
            elif hasattr(gc_policy, "get_action_distribution"):
                baseline_dist = gc_policy.get_action_distribution(o, baseline_goal)
                alt_dist = gc_policy.get_action_distribution(o, alt_goal)
                # Agreement if both distributions peak at same action
                first_action_agreements.append(
                    int(np.argmax(baseline_dist) == np.argmax(alt_dist))
                )

            # Trajectory-level divergence
            h = min(horizon, T - t)
            if h > 0 and hasattr(gc_policy, "predict"):
                n_diverge = 0
                for dt in range(h):
                    a_base = gc_policy.predict(obs[t + dt], subgoals[min(t + dt, T - 1)])
                    a_alt = gc_policy.predict(obs[t + dt], alt_goals[min(t + dt, T - 1)])
                    n_diverge += int(a_base != a_alt)
                traj_divergences.append(n_diverge / h)

            sample_count += 1

        if sample_count >= n_samples:
            break

    first_action_agreement = (
        float(np.mean(first_action_agreements)) if first_action_agreements else 0.0
    )
    mean_traj_divergence = (
        float(np.mean(traj_divergences)) if traj_divergences else 0.0
    )

    return {
        "first_action_agreement": first_action_agreement,
        "mean_trajectory_divergence": mean_traj_divergence,
    }


# ---------------------------------------------------------------------------
# RL metric 3: Goal Necessity – Entropy Correlation
# ---------------------------------------------------------------------------

def compute_gn_entropy_correlation(
    trajectory_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Goal Necessity vs Behavioral Informativeness.

    Bins timesteps by GN quartile and computes the mean entropy
    reduction  H[pi(a|o)] - H[pi(a|o,g)] per quartile.  A good
    explanation should show the highest entropy reduction in the
    high-GN quartile.

    Parameters
    ----------
    trajectory_results : list of dict
        Each dict is expected to have:
        - 'gn_score'         : float  Goal Necessity score for this step
        - 'H_unconditional'  : float  H[pi(a|o)]
        - 'H_conditional'    : float  H[pi(a|o,g)]

    Returns
    -------
    dict with keys:
        'quartile_entropy_reduction' : list of float (length 4, Q1 .. Q4)
        'monotone_quartile'          : bool
        'spearman_rho'               : float
        'p_value'                    : float
    """
    from scipy.stats import spearmanr

    gn_scores = np.array([r["gn_score"] for r in trajectory_results])
    h_uncond = np.array([r["H_unconditional"] for r in trajectory_results])
    h_cond = np.array([r["H_conditional"] for r in trajectory_results])

    entropy_reduction = h_uncond - h_cond

    # Bin into quartiles
    quartile_er: List[float] = []
    boundaries = np.percentile(gn_scores, [25, 50, 75, 100])
    lower = -np.inf
    for upper in boundaries:
        mask = (gn_scores > lower) & (gn_scores <= upper)
        if mask.sum() > 0:
            quartile_er.append(float(entropy_reduction[mask].mean()))
        else:
            quartile_er.append(0.0)
        lower = upper

    # Is entropy reduction monotonically increasing across quartiles?
    monotone = all(quartile_er[i] <= quartile_er[i + 1] for i in range(len(quartile_er) - 1))

    # Spearman correlation between GN and entropy reduction
    if len(gn_scores) >= 3:
        rho, pval = spearmanr(gn_scores, entropy_reduction)
    else:
        rho, pval = 0.0, 1.0

    return {
        "quartile_entropy_reduction": quartile_er,
        "monotone_quartile": bool(monotone),
        "spearman_rho": float(rho),
        "p_value": float(pval),
    }


# ---------------------------------------------------------------------------
# RL metric 4: Human-Proxy Predictability
# ---------------------------------------------------------------------------

def compute_human_proxy_predictability(
    trajectories: List[Dict[str, Any]],
    policy: Any,
    gc_policy: Any,
    jacobian_saliency: np.ndarray,
    subgoal_assignments: List[int],
    gn_scores: np.ndarray,
    horizon: int = 5,
    hidden_sizes: Tuple[int, ...] = (64, 64),
    n_subgoals: int = 10,
    n_actions: int = 6,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Human-Proxy Predictability via 2-layer MLP.

    Trains four MLPs on held-out trajectory data to predict the next
    *horizon* actions given different levels of explanatory information:

    (a) obs only
    (b) obs + Jacobian saliency (flattened)
    (c) obs + one-hot subgoal
    (d) obs + one-hot subgoal + GN score

    Returns the 5-step action prediction accuracy for each condition.

    Parameters
    ----------
    trajectories : list of dict
        Each dict has 'obs' ``(T, obs_dim)`` and 'actions' ``(T,)`` int.
    policy : Any
        Unused (kept for API consistency); can be None.
    gc_policy : Any
        Unused (kept for API consistency); can be None.
    jacobian_saliency : np.ndarray
        Shape ``(N_total_steps, obs_dim)`` – saliency for each timestep.
    subgoal_assignments : list of int
        Length N_total_steps, sub-goal index per step.
    gn_scores : np.ndarray
        Shape ``(N_total_steps,)`` Goal Necessity scores.
    horizon : int
        Number of future actions to predict (default 5).
    hidden_sizes : tuple of int
        Hidden layer sizes for the MLP (default (64, 64)).
    n_subgoals : int
        Cardinality of subgoal vocabulary (for one-hot).
    n_actions : int
        Number of discrete actions in the environment (default 6).
    random_state : int
        Random seed for train/test split and initialisation.

    Returns
    -------
    dict with keys:
        'acc_obs_only'       : float
        'acc_obs_saliency'   : float
        'acc_obs_subgoal'    : float
        'acc_obs_subgoal_gn' : float
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputClassifier

    # Build flat dataset
    all_obs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []  # shape (horizon,) each

    step_idx = 0
    for traj in trajectories:
        obs = np.array(traj["obs"])    # (T, obs_dim)
        acts = np.array(traj["actions"])  # (T,)
        T = len(acts)
        for t in range(T - horizon):
            all_obs.append(obs[t].flatten())
            all_actions.append(acts[t : t + horizon])
            step_idx += 1

    N = len(all_obs)
    if N == 0:
        return {
            "acc_obs_only": 0.0,
            "acc_obs_saliency": 0.0,
            "acc_obs_subgoal": 0.0,
            "acc_obs_subgoal_gn": 0.0,
        }

    obs_arr = np.array(all_obs)           # (N, obs_dim)
    acts_arr = np.array(all_actions)      # (N, horizon)
    obs_dim = obs_arr.shape[1]

    # Align auxiliary features (may be shorter if trajectories differ)
    # Truncate saliency / subgoals / gn to N
    if jacobian_saliency is not None and len(jacobian_saliency) >= N:
        sal_arr = jacobian_saliency[:N].reshape(N, -1)
    else:
        sal_arr = np.zeros((N, obs_dim), dtype=np.float32)

    if subgoal_assignments is not None and len(subgoal_assignments) >= N:
        sg_arr = np.array(subgoal_assignments[:N], dtype=int)
    else:
        sg_arr = np.zeros(N, dtype=int)

    if gn_scores is not None and len(gn_scores) >= N:
        gn_arr = np.array(gn_scores[:N], dtype=np.float32).reshape(-1, 1)
    else:
        gn_arr = np.zeros((N, 1), dtype=np.float32)

    # One-hot subgoal
    sg_onehot = np.zeros((N, n_subgoals), dtype=np.float32)
    for i, sg in enumerate(sg_arr):
        sg_onehot[i, min(sg, n_subgoals - 1)] = 1.0

    # Build feature matrices for the four conditions
    feature_sets = {
        "acc_obs_only":       obs_arr,
        "acc_obs_saliency":   np.concatenate([obs_arr, sal_arr], axis=1),
        "acc_obs_subgoal":    np.concatenate([obs_arr, sg_onehot], axis=1),
        "acc_obs_subgoal_gn": np.concatenate([obs_arr, sg_onehot, gn_arr], axis=1),
    }

    results: Dict[str, float] = {}
    rng = np.random.default_rng(random_state)

    for key, X in feature_sets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, acts_arr, test_size=0.2, random_state=random_state
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            max_iter=200,
            random_state=random_state,
        )

        if horizon == 1:
            mlp.fit(X_train, y_train.ravel())
            acc = mlp.score(X_test, y_test.ravel())
        else:
            # Multi-output: predict each of the horizon actions independently
            mo_clf = MultiOutputClassifier(mlp)
            mo_clf.fit(X_train, y_train)
            y_pred = np.array(mo_clf.predict(X_test))  # (horizon, N_test)
            if y_pred.ndim == 2 and y_pred.shape[0] == horizon:
                y_pred = y_pred.T  # -> (N_test, horizon)
            acc = float((y_pred == y_test).all(axis=1).mean())

        results[key] = float(acc)

    return results


# ---------------------------------------------------------------------------
# Helper: resize map
# ---------------------------------------------------------------------------

def _resize_map(arr: np.ndarray, size: int) -> np.ndarray:
    """Bilinearly resize a 2-D array to ``(size, size)`` using PyTorch."""
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    t_up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t_up.squeeze().numpy()
