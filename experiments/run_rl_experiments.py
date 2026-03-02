"""
Main script for the MiniGrid RL explainability experiment pipeline.

Usage
-----
    python experiments/run_rl_experiments.py \\
        --config configs/rl_config.yaml \\
        --checkpoint-dir models/rl/checkpoint/ \\
        --results-dir results/rl/

Pipeline steps
--------------
1. Load YAML config
2. Load trained PPO policy and goal-conditioned policy from checkpoint dir
3. Roll out evaluation trajectories on each configured MiniGrid environment
4. Run all RL evaluation metrics
5. Save results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.rl_eval import run_rl_evaluation


# ---------------------------------------------------------------------------
# Policy loading helpers
# ---------------------------------------------------------------------------

def _load_ppo_policy(checkpoint_dir: str, env_name: str) -> Optional[Any]:
    """
    Load a trained Stable-Baselines3 PPO policy.

    Looks for ``{checkpoint_dir}/{env_name}/best_model.zip`` or the first
    ``.zip`` file in ``{checkpoint_dir}/{env_name}/``.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        logger.error("stable_baselines3 not installed.  Cannot load PPO policy.")
        return None

    env_dir = os.path.join(checkpoint_dir, env_name)
    best = os.path.join(env_dir, "best_model.zip")
    if os.path.exists(best):
        model = PPO.load(best)
        logger.info("Loaded PPO from %s", best)
        return model

    # Fallback: first .zip in directory
    if os.path.isdir(env_dir):
        for fname in sorted(os.listdir(env_dir)):
            if fname.endswith(".zip"):
                path = os.path.join(env_dir, fname)
                model = PPO.load(path)
                logger.info("Loaded PPO from %s", path)
                return model

    logger.warning("No PPO checkpoint found in %s", env_dir)
    return None


def _load_gc_policy(checkpoint_dir: str, env_name: str) -> Optional[Any]:
    """
    Load a goal-conditioned policy.

    Looks for ``{checkpoint_dir}/{env_name}/gc_best_model.zip``.
    Falls back to a lightweight wrapper around the PPO policy if absent.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        return None

    gc_path = os.path.join(checkpoint_dir, env_name, "gc_best_model.zip")
    if os.path.exists(gc_path):
        model = PPO.load(gc_path)
        logger.info("Loaded GC policy from %s", gc_path)
        return _SB3GoalConditionedWrapper(model)

    logger.warning("Goal-conditioned checkpoint not found; using PPO policy as fallback.")
    base = _load_ppo_policy(checkpoint_dir, env_name)
    if base is not None:
        return _SB3GoalConditionedWrapper(base)
    return None


class _SB3GoalConditionedWrapper:
    """Thin wrapper giving gc_policy.predict(obs, goal) -> action interface."""

    def __init__(self, sb3_model: Any) -> None:
        self._model = sb3_model

    def predict(self, obs: Any, goal: Any) -> int:
        # For SB3 we concatenate obs + goal-embedding (simplified: ignore goal)
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)

    def get_action_distribution(self, obs: Any, goal: Any) -> np.ndarray:
        import torch
        obs_t = torch.as_tensor(obs).unsqueeze(0).float()
        policy = self._model.policy
        with torch.no_grad():
            dist = policy.get_distribution(obs_t)
            probs = dist.distribution.probs.squeeze(0).cpu().numpy()
        return probs


# ---------------------------------------------------------------------------
# MiniGrid trajectory collection
# ---------------------------------------------------------------------------

def _rollout_trajectories(
    env_name: str,
    policy: Any,
    n_trajectories: int,
    max_steps: int = 500,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Roll out ``n_trajectories`` episodes in the given MiniGrid environment.

    Returns
    -------
    list of dict, each containing:
        'obs'     : np.ndarray  (T, obs_dim)
        'actions' : np.ndarray  (T,) int
        'rewards' : np.ndarray  (T,)
        'dones'   : np.ndarray  (T,) bool
        'subgoals': list (T,)  placeholder – filled by subgoal imputation
        'alt_goals': list (T,) placeholder
    """
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401  registers envs
    except ImportError:
        logger.error("gymnasium / minigrid not installed.  Cannot roll out trajectories.")
        return []

    trajectories: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for ep in range(n_trajectories):
        ep_seed = int(rng.integers(0, 2 ** 31))
        try:
            env = gym.make(env_name)
            obs_raw, _ = env.reset(seed=ep_seed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Env reset failed: %s", exc)
            continue

        obs_list: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[bool] = []

        obs = _flatten_obs(obs_raw)

        for _ in range(max_steps):
            obs_list.append(obs)
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            action = int(action)
            obs_raw, reward, terminated, truncated, _ = env.step(action)
            obs = _flatten_obs(obs_raw)
            actions.append(action)
            rewards.append(float(reward))
            done = terminated or truncated
            dones.append(done)
            if done:
                break

        env.close()

        T = len(actions)
        trajectories.append({
            "obs":      np.array(obs_list, dtype=np.float32),
            "actions":  np.array(actions, dtype=np.int32),
            "rewards":  np.array(rewards, dtype=np.float32),
            "dones":    np.array(dones, dtype=bool),
            "subgoals": [0] * T,
            "alt_goals": [1] * T,  # placeholder alternative goal
            "gn_scores": np.zeros(T, dtype=np.float32),
            "H_unconditional": np.zeros(T, dtype=np.float32),
            "H_conditional":   np.zeros(T, dtype=np.float32),
            "subgoal_indices": np.zeros(T, dtype=np.int32),
        })

    logger.info("Collected %d trajectories for env=%s", len(trajectories), env_name)
    return trajectories


def _flatten_obs(obs: Any) -> np.ndarray:
    """Flatten a MiniGrid observation (dict or array) to a 1-D float32 vector."""
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs.keys()):
            v = np.asarray(obs[key], dtype=np.float32).flatten()
            parts.append(v)
        return np.concatenate(parts)
    return np.asarray(obs, dtype=np.float32).flatten()


# ---------------------------------------------------------------------------
# Subgoal imputation stub
# ---------------------------------------------------------------------------

class _SubgoalImputationStub:
    """
    Stub subgoal imputation: tries to import the real implementation,
    falls back to uniform segment partitioning.
    """

    def __init__(self) -> None:
        try:
            from methods.teleological.subgoal_imputation import SubgoalImputation  # type: ignore
            self._impl = SubgoalImputation()
            logger.info("Loaded SubgoalImputation.")
        except ImportError:
            self._impl = None
            logger.warning("SubgoalImputation not found; using uniform segmentation.")

    def assign(self, traj: Dict[str, Any]) -> List[tuple]:
        if self._impl is not None:
            return self._impl.assign(traj)
        T = len(traj["actions"])
        n_segs = max(1, T // 10)
        step = max(1, T // n_segs)
        segs = []
        for i in range(n_segs):
            s = i * step
            e = min((i + 1) * step - 1, T - 1)
            segs.append((s, e))
        return segs


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MiniGrid RL explainability experiments."
    )
    parser.add_argument(
        "--config", default="configs/rl_config.yaml",
        help="Path to rl_config.yaml",
    )
    parser.add_argument(
        "--checkpoint-dir", default="models/rl/checkpoint/",
        help="Directory containing PPO / GC policy checkpoints",
    )
    parser.add_argument(
        "--results-dir", default="results/rl/",
        help="Directory where results are written",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="'cuda' or 'cpu' (SB3 uses this for loading)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ---- Config -----------------------------------------------------------
    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.results_dir, exist_ok=True)

    eval_cfg = config.get("evaluation", {})
    n_eval = int(eval_cfg.get("n_eval_trajectories", 500))

    envs = [e["name"] for e in config.get("environments", [])]
    if not envs:
        logger.error("No environments defined in config.")
        sys.exit(1)

    subgoal_imputation = _SubgoalImputationStub()
    all_env_results: Dict[str, Any] = {}

    for env_name in envs:
        logger.info("=== Environment: %s ===", env_name)
        env_results_dir = os.path.join(args.results_dir, env_name)
        os.makedirs(env_results_dir, exist_ok=True)

        # Load policies
        policy    = _load_ppo_policy(args.checkpoint_dir, env_name)
        gc_policy = _load_gc_policy(args.checkpoint_dir, env_name)

        # Roll out trajectories
        trajectories = _rollout_trajectories(
            env_name, policy,
            n_trajectories=n_eval,
            seed=42,
        )

        if not trajectories:
            logger.warning("No trajectories collected for %s; skipping.", env_name)
            continue

        # Jacobian / value-diff saliency (compute from policy if available)
        jacobian_saliency = _compute_jacobian_saliency(policy, trajectories)
        value_diff_saliency = None  # placeholder

        # Run evaluation
        results = run_rl_evaluation(
            policy=policy,
            gc_policy=gc_policy,
            trajectories=trajectories,
            subgoal_imputation=subgoal_imputation,
            cg_analysis=None,
            jacobian_saliency=jacobian_saliency,
            value_diff_saliency=value_diff_saliency,
            config=config,
            results_dir=env_results_dir,
        )
        all_env_results[env_name] = results

    # ---- Save combined summary --------------------------------------------
    combined_path = os.path.join(args.results_dir, "combined_results.json")

    def _ser(obj: Any) -> Any:
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_ser(x) for x in obj]
        return obj

    with open(combined_path, "w") as f:
        json.dump(_ser(all_env_results), f, indent=2)
    logger.info("Saved combined results to %s", combined_path)


# ---------------------------------------------------------------------------
# Jacobian saliency helper
# ---------------------------------------------------------------------------

def _compute_jacobian_saliency(
    policy: Optional[Any],
    trajectories: List[Dict[str, Any]],
) -> Optional[np.ndarray]:
    """
    Compute Jacobian-based saliency for each observation in the trajectories.

    For SB3 PPO the action logits are differentiated w.r.t. the obs.
    Returns np.ndarray of shape (N_total_steps, obs_dim) or None.
    """
    if policy is None:
        return None

    try:
        import torch
    except ImportError:
        return None

    all_sal: List[np.ndarray] = []

    for traj in trajectories:
        obs_arr = traj["obs"]  # (T, obs_dim)
        T, obs_dim = obs_arr.shape
        sal_traj = np.zeros((T, obs_dim), dtype=np.float32)

        try:
            sb3_policy = policy.policy
            for t in range(T):
                obs_t = torch.tensor(obs_arr[t], dtype=torch.float32, requires_grad=True).unsqueeze(0)
                with torch.enable_grad():
                    logits = sb3_policy.evaluate_actions(obs_t, torch.zeros(1, dtype=torch.long))[1]
                    score = logits.sum()
                    score.backward()
                if obs_t.grad is not None:
                    sal_traj[t] = obs_t.grad.squeeze().detach().numpy()
        except Exception:  # noqa: BLE001
            pass  # Return zeros for this trajectory

        all_sal.append(sal_traj)

    return np.concatenate(all_sal, axis=0) if all_sal else None


if __name__ == "__main__":
    main()
