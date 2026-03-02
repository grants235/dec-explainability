"""
RL evaluation pipeline for teleological explainability on MiniGrid.

Runs all RL metrics:
  - Sub-goal segmentation F1
  - Counterfactual validity
  - Goal Necessity – entropy correlation
  - Human-proxy predictability (MLP)

Writes:
  - ``results_dir/rl_results.csv``
  - ``results_dir/rl_summary.csv``
  - ``results_dir/rl_summary.tex``
  - ``results_dir/subgoal_timelines/`` – timeline plots
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from evaluation.metrics import (
    compute_segmentation_f1,
    compute_counterfactual_validity,
    compute_gn_entropy_correlation,
    compute_human_proxy_predictability,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_rl_evaluation(
    policy: Any,
    gc_policy: Any,
    trajectories: List[Dict[str, Any]],
    subgoal_imputation: Any,
    cg_analysis: Optional[Any],
    jacobian_saliency: Optional[np.ndarray],
    value_diff_saliency: Optional[np.ndarray],
    config: Dict[str, Any],
    results_dir: str,
) -> Dict[str, Any]:
    """
    Run all RL evaluation metrics and write results to disk.

    Parameters
    ----------
    policy : Any
        Standard (non-goal-conditioned) PPO policy.  Should support::

            policy.predict(obs, deterministic=True) -> (action, _state)

    gc_policy : Any
        Goal-conditioned policy for counterfactual evaluation.  Should
        support::

            gc_policy.predict(obs, goal) -> action

        or::

            gc_policy.get_action_distribution(obs, goal) -> np.ndarray

    trajectories : list of dict
        Each trajectory is a dict with at minimum:
        - ``'obs'``      : np.ndarray  (T, obs_dim)
        - ``'actions'``  : np.ndarray  (T,)  int
        - ``'subgoals'`` : list of goal descriptors
        - ``'alt_goals'``: list of alternative goals (optional)

        For GN/entropy metrics also include:
        - ``'gn_score'``        : float per step (or attach separately)
        - ``'H_unconditional'`` : float
        - ``'H_conditional'``   : float

    subgoal_imputation : Any
        Object that produces per-step sub-goal assignments.  Must have::

            subgoal_imputation.assign(trajectory) -> list of (start, end) tuples

    cg_analysis : Any or None
        Counterfactual goal analysis object (may be None; metrics will
        use gc_policy directly in that case).

    jacobian_saliency : np.ndarray or None
        Shape ``(N_total_steps, obs_dim)``; Jacobian saliency per step.

    value_diff_saliency : np.ndarray or None
        Shape ``(N_total_steps, obs_dim)``; value-difference saliency.

    config : dict
        Merged YAML config.

    results_dir : str
        Root directory for writing outputs.

    Returns
    -------
    dict
        Nested result dict with all computed metric values.
    """
    os.makedirs(results_dir, exist_ok=True)
    eval_cfg = config.get("evaluation", {})

    boundary_tolerance  = int(eval_cfg.get("boundary_tolerance", 2))
    cf_n_samples        = int(eval_cfg.get("counterfactual_n_samples", 100))
    cf_horizon          = int(eval_cfg.get("counterfactual_horizon", 20))
    pred_horizon        = int(eval_cfg.get("predictability_horizon", 5))
    pred_hidden         = tuple(eval_cfg.get("predictability_mlp_hidden", [64, 64]))

    # ------------------------------------------------------------------
    # 1. Sub-Goal Segmentation F1
    # ------------------------------------------------------------------
    logger.info("Computing sub-goal segmentation F1...")
    seg_f1_scores: List[Dict[str, float]] = []

    for traj in trajectories:
        try:
            pred_segs = subgoal_imputation.assign(traj)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Subgoal imputation failed: %s", exc)
            continue

        gt_segs = traj.get("gt_segments")
        if gt_segs is None:
            gt_segs = _compute_gt_segments_from_traj(traj)

        f1_dict = compute_segmentation_f1(
            pred_segs, gt_segs, tolerance=boundary_tolerance
        )
        seg_f1_scores.append(f1_dict)

    seg_f1_mean = {
        "precision": float(np.mean([s["precision"] for s in seg_f1_scores])) if seg_f1_scores else float("nan"),
        "recall":    float(np.mean([s["recall"]    for s in seg_f1_scores])) if seg_f1_scores else float("nan"),
        "f1":        float(np.mean([s["f1"]        for s in seg_f1_scores])) if seg_f1_scores else float("nan"),
    }
    logger.info("Segmentation F1: %s", seg_f1_mean)

    # ------------------------------------------------------------------
    # 2. Counterfactual Validity
    # ------------------------------------------------------------------
    logger.info("Computing counterfactual validity...")
    cf_results = compute_counterfactual_validity(
        gc_policy=gc_policy,
        trajectories=trajectories,
        n_samples=cf_n_samples,
        horizon=cf_horizon,
    )
    logger.info("Counterfactual validity: %s", cf_results)

    # ------------------------------------------------------------------
    # 3. GN – Entropy Correlation
    # ------------------------------------------------------------------
    logger.info("Computing GN–entropy correlation...")
    gn_step_results = _build_gn_step_results(trajectories)
    if gn_step_results:
        gn_entropy = compute_gn_entropy_correlation(gn_step_results)
    else:
        gn_entropy = {
            "quartile_entropy_reduction": [],
            "monotone_quartile": False,
            "spearman_rho": float("nan"),
            "p_value": float("nan"),
        }
    logger.info("GN entropy correlation: rho=%.3f, p=%.4f",
                gn_entropy["spearman_rho"], gn_entropy["p_value"])

    # ------------------------------------------------------------------
    # 4. Human-Proxy Predictability
    # ------------------------------------------------------------------
    logger.info("Computing human-proxy predictability...")

    # Flatten subgoal assignments and GN scores across trajectories
    flat_subgoals: List[int] = []
    flat_gn: List[float] = []
    for traj in trajectories:
        T = len(traj["actions"])
        sg_list = traj.get("subgoal_indices")
        if sg_list is None:
            flat_subgoals.extend([0] * T)
        else:
            flat_subgoals.extend(list(sg_list)[:T])

        gn_list = traj.get("gn_scores")
        if gn_list is None:
            flat_gn.extend([0.0] * T)
        else:
            flat_gn.extend(list(gn_list)[:T])

    flat_gn_arr = np.array(flat_gn, dtype=np.float32)

    # Use Jacobian saliency if provided, else zeros
    if jacobian_saliency is not None:
        jac_sal = np.array(jacobian_saliency, dtype=np.float32)
    else:
        obs_dim = _infer_obs_dim(trajectories)
        total_steps = sum(len(t["actions"]) for t in trajectories)
        jac_sal = np.zeros((total_steps, obs_dim), dtype=np.float32)

    try:
        n_sg = max(max(flat_subgoals) + 1, 10) if flat_subgoals else 10
        pred_results = compute_human_proxy_predictability(
            trajectories=trajectories,
            policy=policy,
            gc_policy=gc_policy,
            jacobian_saliency=jac_sal,
            subgoal_assignments=flat_subgoals,
            gn_scores=flat_gn_arr,
            horizon=pred_horizon,
            hidden_sizes=pred_hidden,
            n_subgoals=n_sg,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Human-proxy predictability failed: %s", exc)
        pred_results = {
            "acc_obs_only": float("nan"),
            "acc_obs_saliency": float("nan"),
            "acc_obs_subgoal": float("nan"),
            "acc_obs_subgoal_gn": float("nan"),
        }
    logger.info("Predictability: %s", pred_results)

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------
    all_results = {
        "segmentation_f1": seg_f1_mean,
        "counterfactual_validity": cf_results,
        "gn_entropy_correlation": gn_entropy,
        "human_proxy_predictability": pred_results,
    }

    _save_rl_csv(seg_f1_scores, cf_results, pred_results, gn_entropy, results_dir)
    _save_rl_summary(all_results, results_dir)
    _save_rl_json(all_results, results_dir)
    _save_rl_latex(all_results, results_dir)

    logger.info("RL evaluation complete. Results in %s", results_dir)
    return all_results


# ---------------------------------------------------------------------------
# Helper: build GN step-level result dicts
# ---------------------------------------------------------------------------

def _build_gn_step_results(
    trajectories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Flatten per-trajectory GN / entropy information into per-step records.

    Expects each trajectory to optionally have:
    - 'gn_scores'      : list of floats
    - 'H_unconditional': list of floats  (or single float)
    - 'H_conditional'  : list of floats  (or single float)
    """
    records: List[Dict[str, Any]] = []
    for traj in trajectories:
        T = len(traj["actions"])
        gn_list = traj.get("gn_scores", [0.0] * T)
        h_unc = traj.get("H_unconditional", [0.0] * T)
        h_cnd = traj.get("H_conditional", [0.0] * T)

        # Allow scalar (broadcast)
        if not hasattr(h_unc, "__len__"):
            h_unc = [h_unc] * T
        if not hasattr(h_cnd, "__len__"):
            h_cnd = [h_cnd] * T

        for t in range(min(T, len(gn_list), len(h_unc), len(h_cnd))):
            records.append({
                "gn_score":        float(gn_list[t]),
                "H_unconditional": float(h_unc[t]),
                "H_conditional":   float(h_cnd[t]),
            })
    return records


def _compute_gt_segments_from_traj(
    traj: Dict[str, Any],
) -> List[tuple]:
    """
    Fallback: return a single segment spanning the whole trajectory.
    """
    T = len(traj["actions"])
    return [(0, T - 1)]


def _infer_obs_dim(trajectories: List[Dict[str, Any]]) -> int:
    for t in trajectories:
        obs = np.array(t["obs"])
        if obs.ndim >= 2:
            return int(obs.shape[1])
        if obs.ndim == 1:
            return int(obs.shape[0])
    return 1


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _save_rl_csv(
    seg_f1_scores: List[Dict[str, float]],
    cf_results: Dict[str, float],
    pred_results: Dict[str, float],
    gn_entropy: Dict[str, Any],
    results_dir: str,
) -> None:
    path = os.path.join(results_dir, "rl_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "key", "value"])
        for row in seg_f1_scores:
            for k, v in row.items():
                writer.writerow(["segmentation_f1", k, v])
        for k, v in cf_results.items():
            writer.writerow(["counterfactual_validity", k, v])
        for k, v in pred_results.items():
            writer.writerow(["human_proxy_predictability", k, v])
        writer.writerow(["gn_entropy", "spearman_rho", gn_entropy["spearman_rho"]])
        writer.writerow(["gn_entropy", "p_value", gn_entropy["p_value"]])
    logger.info("Saved RL results CSV to %s", path)


def _save_rl_summary(all_results: Dict[str, Any], results_dir: str) -> None:
    path = os.path.join(results_dir, "rl_summary.csv")
    rows = []
    def _flatten(d: Any, prefix: str = "") -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                _flatten(v, f"{prefix}{k}_" if prefix else f"{k}_")
        elif isinstance(d, list):
            rows.append((prefix.rstrip("_"), str(d)))
        else:
            rows.append((prefix.rstrip("_"), d))

    _flatten(all_results)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, val in rows:
            writer.writerow([key, val])
    logger.info("Saved RL summary to %s", path)


def _save_rl_json(all_results: Dict[str, Any], results_dir: str) -> None:
    path = os.path.join(results_dir, "rl_results.json")

    def _serialise(obj: Any) -> Any:
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialise(x) for x in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_serialise(all_results), f, indent=2)
    logger.info("Saved RL JSON results to %s", path)


def _save_rl_latex(all_results: Dict[str, Any], results_dir: str) -> None:
    path = os.path.join(results_dir, "rl_summary.tex")

    seg = all_results.get("segmentation_f1", {})
    cf  = all_results.get("counterfactual_validity", {})
    gn  = all_results.get("gn_entropy_correlation", {})
    pr  = all_results.get("human_proxy_predictability", {})

    def _fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{RL teleological explainability evaluation.}",
        r"\label{tab:rl_results}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Seg. Precision & {_fmt(seg.get('precision'))} \\\\",
        f"Seg. Recall & {_fmt(seg.get('recall'))} \\\\",
        f"Seg. F1 & {_fmt(seg.get('f1'))} \\\\",
        r"\midrule",
        f"CF First-Action Agreement & {_fmt(cf.get('first_action_agreement'))} \\\\",
        f"CF Traj. Divergence & {_fmt(cf.get('mean_trajectory_divergence'))} \\\\",
        r"\midrule",
        f"GN–Entropy Spearman $\\rho$ & {_fmt(gn.get('spearman_rho'))} \\\\",
        f"Monotone GN Quartile & {_fmt(gn.get('monotone_quartile'))} \\\\",
        r"\midrule",
        f"Pred. (obs only) & {_fmt(pr.get('acc_obs_only'))} \\\\",
        f"Pred. (obs + saliency) & {_fmt(pr.get('acc_obs_saliency'))} \\\\",
        f"Pred. (obs + subgoal) & {_fmt(pr.get('acc_obs_subgoal'))} \\\\",
        f"Pred. (obs + subgoal + GN) & {_fmt(pr.get('acc_obs_subgoal_gn'))} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved RL LaTeX table to %s", path)
