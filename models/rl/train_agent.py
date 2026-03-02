"""
Train PPO agents on MiniGrid environments using stable-baselines3.

Supports three environments:
  - MiniGrid-DoorKey-8x8-v0       (2M steps)
  - MiniGrid-KeyCorridorS4R3-v0   (5M steps)
  - MiniGrid-MultiRoom-N6-v0      (5M steps)

After training, collects 500 evaluation trajectories per environment storing
observations, actions, rewards, grid states, policy logits, and value estimates.

CLI usage:
    python train_agent.py --env MiniGrid-DoorKey-8x8-v0
    python train_agent.py --env MiniGrid-DoorKey-8x8-v0 --collect-only
    python train_agent.py --env MiniGrid-DoorKey-8x8-v0 --save-dir /path/to/dir
    python train_agent.py --env MiniGrid-DoorKey-8x8-v0 --timesteps 1000000
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from tqdm import tqdm

# MiniGrid wrappers
import minigrid
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_TIMESTEPS: Dict[str, int] = {
    "MiniGrid-DoorKey-8x8-v0": 2_000_000,
    "MiniGrid-KeyCorridorS4R3-v0": 5_000_000,
    "MiniGrid-MultiRoom-N6-v0": 5_000_000,
}

N_EVAL_TRAJECTORIES = 500

# Default save directory (relative to this script's location)
_SCRIPT_DIR = Path(__file__).parent
DEFAULT_SAVE_DIR = _SCRIPT_DIR / "checkpoint"


# ---------------------------------------------------------------------------
# Custom CNN Feature Extractor
# ---------------------------------------------------------------------------

class MiniGridCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for MiniGrid image observations.

    Architecture:
        Conv(3->16, 3x3, stride=2) -> ReLU
        Conv(16->32, 3x3, stride=2) -> ReLU
        Conv(32->64, 3x3, stride=1) -> ReLU
        Flatten
        FC(flattened_size -> 256) -> ReLU

    The SB3 ActorCriticPolicy will add separate policy (256->n_actions)
    and value (256->1) heads on top of the 256-d features.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)

        # MiniGrid fully-observed image: (H, W, 3) -> rearranged to (3, H, W)
        n_input_channels = observation_space.shape[2]  # channel-last from MiniGrid

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened CNN output size dynamically
        with torch.no_grad():
            # observation_space.shape is (H, W, C); transpose for CNN
            dummy = torch.zeros(1, n_input_channels,
                                observation_space.shape[0],
                                observation_space.shape[1])
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 passes (B, H, W, C); permute to (B, C, H, W)
        x = observations.float() / 255.0 if observations.dtype == torch.uint8 else observations.float()
        if x.ndim == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        return self.fc(self.cnn(x))


# ---------------------------------------------------------------------------
# Policy kwargs
# ---------------------------------------------------------------------------

POLICY_KWARGS: Dict[str, Any] = {
    "features_extractor_class": MiniGridCNN,
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": [],  # No extra layers between features and heads; heads are linear
}


# ---------------------------------------------------------------------------
# Environment construction helpers
# ---------------------------------------------------------------------------

def make_env(env_name: str, seed: int = 0):
    """Factory: returns a callable that builds a wrapped MiniGrid env."""
    def _init() -> gym.Env:
        env = gym.make(env_name)
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


def make_vec_env(env_name: str, n_envs: int = 8, seed: int = 0) -> VecMonitor:
    """Build a vectorized (DummyVecEnv) + monitored environment."""
    fns = [make_env(env_name, seed=seed + i) for i in range(n_envs)]
    vec_env = DummyVecEnv(fns)
    vec_env = VecMonitor(vec_env)
    return vec_env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    env_name: str,
    total_timesteps: int,
    save_dir: Path,
    n_envs: int = 8,
    seed: int = 42,
) -> PPO:
    """
    Train a PPO agent on the given MiniGrid environment.

    Parameters
    ----------
    env_name : str
        Gymnasium ID of the MiniGrid environment.
    total_timesteps : int
        Total environment steps to train for.
    save_dir : Path
        Directory where the final checkpoint (.zip) will be saved.
    n_envs : int
        Number of parallel environments.
    seed : int
        Random seed.

    Returns
    -------
    PPO
        The trained SB3 PPO model.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"{env_name}.zip"

    print(f"[train] Environment   : {env_name}")
    print(f"[train] Total steps   : {total_timesteps:,}")
    print(f"[train] Save path     : {checkpoint_path}")
    print(f"[train] n_envs        : {n_envs}")

    vec_env = make_vec_env(env_name, n_envs=n_envs, seed=seed)

    model = PPO(
        policy=ActorCriticPolicy,
        env=vec_env,
        learning_rate=3e-4,
        clip_range=0.2,
        gae_lambda=0.95,
        ent_coef=0.01,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        verbose=1,
        seed=seed,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=str(save_dir / "tb_logs"),
    )

    # Periodic checkpoint callback (saves every 10% of training)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, total_timesteps // (10 * n_envs)),
        save_path=str(save_dir / "intermediate"),
        name_prefix=env_name,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(str(checkpoint_path.with_suffix("")))  # SB3 appends .zip
    print(f"[train] Model saved -> {checkpoint_path}")

    vec_env.close()
    return model


def load_model(checkpoint_path: Path) -> PPO:
    """Load a saved SB3 PPO model from disk."""
    return PPO.load(str(checkpoint_path.with_suffix("")))


# ---------------------------------------------------------------------------
# Trajectory Collection
# ---------------------------------------------------------------------------

def _get_logits_and_value(
    model: PPO, obs: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Run the policy network on a single observation and return
    (policy_logits, value_estimate).

    Parameters
    ----------
    obs : np.ndarray
        A single observation of shape (H, W, C).

    Returns
    -------
    logits : np.ndarray  shape (n_actions,)
    value  : float
    """
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    obs_tensor = obs_tensor.to(model.device)

    with torch.no_grad():
        # SB3 ActorCriticPolicy forward gives (actions, values, log_probs)
        # but we want logits (action distribution parameters) and value separately.
        features = model.policy.extract_features(obs_tensor,
                                                  model.policy.features_extractor)
        latent_pi, latent_vf = model.policy.mlp_extractor(features)

        # Policy distribution parameters (logits for Categorical)
        distribution = model.policy.action_dist.proba_distribution(
            action_logits=model.policy.action_net(latent_pi)
        )
        logits = model.policy.action_net(latent_pi).cpu().numpy().squeeze(0)

        # Value estimate
        value = model.policy.value_net(latent_vf).cpu().item()

    return logits, value


def collect_trajectories(
    env_name: str,
    model: PPO,
    n_trajectories: int = N_EVAL_TRAJECTORIES,
    seed: int = 0,
    max_steps: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Collect evaluation trajectories using a greedy (deterministic) policy.

    For each trajectory, records:
        obs_seq       : list of np.ndarray (H, W, C), raw observations
        action_seq    : list of int
        reward_seq    : list of float
        grid_state_seq: list of np.ndarray (W, H, 3), env.grid.encode() snapshots
        logits_seq    : list of np.ndarray (n_actions,), policy logits
        value_seq     : list of float, value estimates

    Parameters
    ----------
    env_name : str
        Gymnasium environment ID.
    model : PPO
        Trained SB3 PPO model.
    n_trajectories : int
        Number of trajectories to collect.
    seed : int
        Base random seed.
    max_steps : int
        Maximum steps per trajectory before forced termination.

    Returns
    -------
    List of trajectory dicts.
    """
    trajectories: List[Dict[str, Any]] = []

    # Build a single unwrapped env to access grid state
    raw_env = gym.make(env_name)
    raw_env = FullyObsWrapper(raw_env)
    obs_env = ImgObsWrapper(raw_env)

    print(f"[collect] Collecting {n_trajectories} trajectories for {env_name} ...")

    for traj_idx in tqdm(range(n_trajectories), desc="Trajectories"):
        obs, _ = obs_env.reset(seed=seed + traj_idx)

        obs_seq: List[np.ndarray] = []
        action_seq: List[int] = []
        reward_seq: List[float] = []
        grid_state_seq: List[np.ndarray] = []
        logits_seq: List[np.ndarray] = []
        value_seq: List[float] = []

        done = False
        truncated = False
        step = 0

        while not (done or truncated) and step < max_steps:
            # Capture current state
            obs_seq.append(obs.copy())

            # Grid state: accessible via the unwrapped env
            grid_state = obs_env.unwrapped.grid.encode()  # (W, H, 3)
            grid_state_seq.append(grid_state.copy())

            # Policy forward pass: logits and value
            logits, value = _get_logits_and_value(model, obs)
            logits_seq.append(logits)
            value_seq.append(value)

            # Greedy action (deterministic)
            action = int(np.argmax(logits))

            obs, reward, done, truncated, _ = obs_env.step(action)

            action_seq.append(action)
            reward_seq.append(float(reward))
            step += 1

        trajectories.append({
            "env_name": env_name,
            "traj_idx": traj_idx,
            "obs_seq": obs_seq,
            "action_seq": action_seq,
            "reward_seq": reward_seq,
            "grid_state_seq": grid_state_seq,
            "logits_seq": logits_seq,
            "value_seq": value_seq,
            "total_reward": sum(reward_seq),
            "length": step,
            "success": done and sum(reward_seq) > 0,
        })

    obs_env.close()

    success_rate = np.mean([t["success"] for t in trajectories])
    avg_len = np.mean([t["length"] for t in trajectories])
    print(f"[collect] Done. Success rate: {success_rate:.2%}, Avg length: {avg_len:.1f}")

    return trajectories


def save_trajectories(trajectories: List[Dict[str, Any]], save_path: Path) -> None:
    """Pickle trajectory list to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = save_path.stat().st_size / (1024 ** 2)
    print(f"[save] Trajectories -> {save_path}  ({size_mb:.1f} MB)")


def load_trajectories(save_path: Path) -> List[Dict[str, Any]]:
    """Load pickled trajectory list from disk."""
    with open(save_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO agents on MiniGrid and collect evaluation trajectories."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MiniGrid-DoorKey-8x8-v0",
        choices=list(ENV_TIMESTEPS.keys()),
        help="MiniGrid environment ID.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override training timesteps (default: per-env value from ENV_TIMESTEPS).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(DEFAULT_SAVE_DIR),
        help="Directory to save model checkpoints and trajectories.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip training; load existing checkpoint and only collect trajectories.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments for training.",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=N_EVAL_TRAJECTORIES,
        help="Number of evaluation trajectories to collect.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    checkpoint_path = save_dir / f"{args.env}.zip"
    traj_path = save_dir / f"{args.env}_trajectories.pkl"

    total_timesteps = args.timesteps or ENV_TIMESTEPS[args.env]

    if args.collect_only:
        # Load existing model
        if not checkpoint_path.exists():
            print(f"[error] Checkpoint not found: {checkpoint_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[main] Loading checkpoint: {checkpoint_path}")
        model = load_model(checkpoint_path)
    else:
        # Train from scratch
        model = train(
            env_name=args.env,
            total_timesteps=total_timesteps,
            save_dir=save_dir,
            n_envs=args.n_envs,
            seed=args.seed,
        )

    # Collect evaluation trajectories
    trajectories = collect_trajectories(
        env_name=args.env,
        model=model,
        n_trajectories=args.n_trajectories,
        seed=args.seed,
    )

    save_trajectories(trajectories, traj_path)
    print(f"[main] All done for {args.env}.")


if __name__ == "__main__":
    main()
