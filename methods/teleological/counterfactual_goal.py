"""
Counterfactual Goal Analysis for MiniGrid RL agents.

Overview
--------
Trains a goal-conditioned policy (same CNN backbone as the base agent, but
with an additional goal embedding) using Hindsight Experience Replay (HER)
relabelling.  At inference time, computes:

    - pi_current   = pi(a | o, g_current)       current goal distribution
    - pi_g'        = pi(a | o, g')               alternative goal distribution
    - KL(g')       = KL(pi_current || pi_g')     divergence under each alt. goal
    - delta_action(g') = 1  iff  argmax pi_g' != a_t
    - Goal Necessity (GN) = |{g': delta_action=1}| / |G_alt|
    - g_contrast   = highest-KL goal with delta_action=1

Architecture
------------
GoalConditionedPolicy:
    CNN (same 3-layer design as base agent) -> 256-d features
    Embedding(n_goals, 32)                  -> 32-d goal embedding
    Concat(256+32, 288) -> FC(288, 256) -> ReLU
    Policy head: FC(256, n_actions)
    Value head:  FC(256, 1)

Training with HER
-----------------
1. Collect trajectories with the standard PPO policy.
2. Identify sub-goal completion events (via SubGoalImputation).
3. Relabel: for each (t_end, g) event, create an experience with
       goal = g, reward = +1.0
   For randomly chosen g' != g, create negative examples with
       goal = g', reward = -0.1.
4. Train the goal-conditioned policy with PPO (1M steps).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Local import
try:
    from methods.teleological.subgoal_imputation import SubGoal
except ImportError:
    try:
        from subgoal_imputation import SubGoal
    except ImportError:
        # Minimal fallback if run in isolation
        from dataclasses import dataclass as _dc

        @_dc
        class SubGoal:  # type: ignore[no-redef]
            name: str
            category: str = ""

            def __hash__(self):
                return hash(self.name)

            def __eq__(self, other):
                return isinstance(other, SubGoal) and self.name == other.name


# ---------------------------------------------------------------------------
# Goal-Conditioned Policy Architecture
# ---------------------------------------------------------------------------

class GoalConditionedCNN(nn.Module):
    """
    CNN feature extractor identical to MiniGridCNN in train_agent.py.

    Input  : (B, C, H, W)  float32 in [0, 1]
    Output : (B, 256)
    """

    def __init__(self, n_input_channels: int = 3, obs_h: int = 8, obs_w: int = 8) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, n_input_channels, obs_h, obs_w)
            cnn_out = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out, 256),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(x))


class GoalConditionedPolicy(nn.Module):
    """
    Goal-conditioned actor-critic policy.

    Architecture:
        CNN encoder   -> f  in R^256
        Embedding     -> e  in R^32      (goal_idx -> embedding)
        Linear(288, 256) + ReLU -> h     (288 = 256 + 32)
        Policy head   -> logits  in R^n_actions
        Value head    -> value   in R^1

    Parameters
    ----------
    n_goals : int
        Size of the goal vocabulary.
    n_actions : int
        Number of discrete actions (7 for MiniGrid).
    n_input_channels : int
        Number of channels in the observation (3 for ImgObsWrapper).
    obs_h, obs_w : int
        Spatial dimensions of the observation.
    goal_emb_dim : int
        Dimensionality of the goal embedding (default 32).
    """

    def __init__(
        self,
        n_goals: int,
        n_actions: int = 7,
        n_input_channels: int = 3,
        obs_h: int = 8,
        obs_w: int = 8,
        goal_emb_dim: int = 32,
    ) -> None:
        super().__init__()

        self.n_goals = n_goals
        self.n_actions = n_actions
        self.goal_emb_dim = goal_emb_dim

        # CNN backbone
        self.cnn = GoalConditionedCNN(n_input_channels, obs_h, obs_w)

        # Goal embedding
        self.goal_embedding = nn.Embedding(n_goals, goal_emb_dim)

        # Shared trunk after concatenation
        self.shared = nn.Sequential(
            nn.Linear(256 + goal_emb_dim, 256),
            nn.ReLU(),
        )

        # Actor and critic heads
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head  = nn.Linear(256, 1)

    def forward(
        self, obs: torch.Tensor, goal_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        obs : torch.Tensor  shape (B, C, H, W) or (B, H, W, C) – float32
        goal_idx : torch.Tensor  shape (B,) – long

        Returns
        -------
        logits : torch.Tensor  shape (B, n_actions)
        value  : torch.Tensor  shape (B, 1)
        """
        # Normalise and ensure channel-first
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        else:
            obs = obs.float()

        if obs.ndim == 4 and obs.shape[-1] in (1, 3):
            obs = obs.permute(0, 3, 1, 2)

        cnn_feat = self.cnn(obs)                         # (B, 256)
        goal_emb = self.goal_embedding(goal_idx)         # (B, 32)
        combined = torch.cat([cnn_feat, goal_emb], dim=1)  # (B, 288)
        h = self.shared(combined)                        # (B, 256)
        logits = self.policy_head(h)                     # (B, n_actions)
        value  = self.value_head(h)                      # (B, 1)
        return logits, value

    def get_action_dist(
        self, obs: torch.Tensor, goal_idx: torch.Tensor
    ) -> Categorical:
        """Return the action distribution (Categorical) for given obs and goal."""
        logits, _ = self.forward(obs, goal_idx)
        return Categorical(logits=logits)

    def get_log_prob(
        self, obs: torch.Tensor, goal_idx: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log-prob, value, and entropy for a batch of (obs, goal, action).

        Returns
        -------
        log_probs : (B,)
        values    : (B, 1)
        entropy   : (B,)
        """
        logits, values = self.forward(obs, goal_idx)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()


# ---------------------------------------------------------------------------
# HER Experience Replay Buffer (simple, in-memory)
# ---------------------------------------------------------------------------

@dataclass
class HERExperience:
    """Single HER-relabelled experience."""
    obs: np.ndarray          # (H, W, C)
    goal_idx: int
    action: int
    reward: float
    next_obs: np.ndarray     # (H, W, C)
    done: bool


class HERBuffer:
    """
    Hindsight Experience Replay buffer.

    Collects standard trajectories and creates relabelled experiences:
        - Positive: at timestep t_end where sub-goal g was achieved,
          create experience with goal=g, reward=+1.
        - Negative: for n_neg other goals g' != g, reward=-0.1.

    Parameters
    ----------
    subgoal_vocab : List[SubGoal]
    negative_reward : float
    neg_ratio : float
        Fraction of alternative goals to use as negatives per event.
    """

    def __init__(
        self,
        subgoal_vocab: List[SubGoal],
        negative_reward: float = -0.1,
        neg_ratio: float = 0.5,
    ) -> None:
        self.vocab = subgoal_vocab
        self.goal_name_to_idx: Dict[str, int] = {
            sg.name: i for i, sg in enumerate(subgoal_vocab)
        }
        self.negative_reward = negative_reward
        self.neg_ratio = neg_ratio
        self.experiences: List[HERExperience] = []

    def add_trajectory(
        self,
        trajectory: Dict[str, Any],
        segments: List[Tuple[int, int, "SubGoal"]],
    ) -> None:
        """
        Add HER-relabelled experiences from a trajectory.

        Parameters
        ----------
        trajectory : dict
            Must contain obs_seq, action_seq, reward_seq.
        segments : list of (t_start, t_end, SubGoal)
            From SubGoalImputation.impute().
        """
        obs_seq: List[np.ndarray] = trajectory["obs_seq"]
        action_seq: List[int] = trajectory["action_seq"]
        reward_seq: List[float] = trajectory["reward_seq"]
        T = len(action_seq)

        for t_start, t_end, sg in segments:
            if sg is None or sg.name not in self.goal_name_to_idx:
                continue
            goal_idx = self.goal_name_to_idx[sg.name]

            # Positive experience at completion timestep
            if t_end < T:
                obs = obs_seq[t_end]
                next_obs = obs_seq[t_end + 1] if t_end + 1 < len(obs_seq) else obs_seq[t_end]
                self.experiences.append(HERExperience(
                    obs=obs,
                    goal_idx=goal_idx,
                    action=action_seq[t_end],
                    reward=1.0,
                    next_obs=next_obs,
                    done=True,
                ))

            # Negative experiences: sample other goals
            alt_goals = [
                (i, s) for i, s in enumerate(self.vocab)
                if s.name != sg.name
            ]
            n_neg = max(1, int(len(alt_goals) * self.neg_ratio))
            neg_indices = np.random.choice(len(alt_goals), size=min(n_neg, len(alt_goals)),
                                           replace=False)
            for ni in neg_indices:
                alt_idx, _ = alt_goals[ni]
                if t_end < T:
                    obs = obs_seq[t_end]
                    next_obs = obs_seq[t_end + 1] if t_end + 1 < len(obs_seq) else obs_seq[t_end]
                    self.experiences.append(HERExperience(
                        obs=obs,
                        goal_idx=alt_idx,
                        action=action_seq[t_end],
                        reward=self.negative_reward,
                        next_obs=next_obs,
                        done=False,
                    ))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of HER experiences.

        Returns
        -------
        obs, goal_idxs, actions, rewards, next_obs, dones
        Each is a numpy array of the appropriate shape.
        """
        indices = np.random.randint(0, len(self.experiences), size=batch_size)
        batch = [self.experiences[i] for i in indices]

        obs       = np.stack([e.obs      for e in batch])
        goals     = np.array([e.goal_idx for e in batch], dtype=np.int64)
        actions   = np.array([e.action   for e in batch], dtype=np.int64)
        rewards   = np.array([e.reward   for e in batch], dtype=np.float32)
        next_obs  = np.stack([e.next_obs for e in batch])
        dones     = np.array([e.done     for e in batch], dtype=np.float32)
        return obs, goals, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self.experiences)


# ---------------------------------------------------------------------------
# Goal-conditioned policy training (PPO-style, simplified)
# ---------------------------------------------------------------------------

def train_goal_conditioned_policy(
    policy: GoalConditionedPolicy,
    her_buffer: HERBuffer,
    total_steps: int = 1_000_000,
    batch_size: int = 256,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    n_epochs: int = 4,
    device: str = "cpu",
    log_interval: int = 1000,
) -> List[float]:
    """
    Train the goal-conditioned policy using PPO-style updates on HER experiences.

    Since we operate on a static replay buffer (no online collection here),
    this is essentially supervised PPO: we iterate over the buffer in mini-batches,
    compute policy gradient losses with clipping, and update.

    Parameters
    ----------
    policy : GoalConditionedPolicy
    her_buffer : HERBuffer
    total_steps : int
    batch_size : int
    lr : float
    clip_eps : float
    ent_coef : float
    n_epochs : int
    device : str
    log_interval : int

    Returns
    -------
    List[float] – per-update total losses.
    """
    if len(her_buffer) == 0:
        print("[train_gc] HER buffer is empty; skipping training.")
        return []

    policy = policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    losses: List[float] = []

    n_updates = total_steps // batch_size
    print(f"[train_gc] Training goal-conditioned policy: {n_updates} updates, "
          f"batch_size={batch_size}, device={device}")

    for update in range(n_updates):
        obs_np, goals_np, actions_np, rewards_np, _, dones_np = \
            her_buffer.sample(batch_size)

        obs_t     = torch.as_tensor(obs_np,     dtype=torch.float32, device=device)
        goals_t   = torch.as_tensor(goals_np,   dtype=torch.long,    device=device)
        actions_t = torch.as_tensor(actions_np, dtype=torch.long,    device=device)
        rewards_t = torch.as_tensor(rewards_np, dtype=torch.float32, device=device)

        # Old log-probs (detached, for PPO ratio)
        with torch.no_grad():
            logits_old, values_old = policy(obs_t, goals_t)
            dist_old = Categorical(logits=logits_old)
            log_probs_old = dist_old.log_prob(actions_t)

        for _ in range(n_epochs):
            log_probs, values, entropy = policy.get_log_prob(obs_t, goals_t, actions_t)
            values = values.squeeze(-1)

            # Advantage (simple: reward - baseline; no proper GAE here)
            advantages = rewards_t - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO clipped objective
            ratio = torch.exp(log_probs - log_probs_old)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

            # Value loss
            value_loss = F.mse_loss(values, rewards_t)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            total_loss = policy_loss + 0.5 * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        losses.append(total_loss.item())
        if (update + 1) % log_interval == 0:
            print(f"  [update {update+1}/{n_updates}] loss={np.mean(losses[-log_interval:]):.4f}")

    return losses


# ---------------------------------------------------------------------------
# Counterfactual Goal Analysis
# ---------------------------------------------------------------------------

@dataclass
class TimestepAnalysis:
    """Per-timestep counterfactual analysis result."""
    t: int
    current_goal: SubGoal
    action_taken: int
    kl_scores: Dict[str, float]       # goal_name -> KL divergence
    alt_actions: Dict[str, int]        # goal_name -> argmax action under that goal
    delta_actions: Dict[str, int]      # goal_name -> 0/1 flag
    goal_necessity: float              # GN in [0, 1]
    contrastive_goal: Optional[str]    # name of highest-KL goal with delta=1
    explanation_text: str


class CounterfactualGoalAnalysis:
    """
    Counterfactual Goal Analysis for a trained goal-conditioned policy.

    At inference, for each timestep:
        1. Compute pi_current = pi(a | o, g_current)
        2. For each alt goal g': pi_g' = pi(a | o, g')
        3. KL(g') = KL(pi_current || pi_g')
        4. delta_action(g') = 1  iff  argmax(pi_g') != a_t
        5. GN = |{g': delta_action=1}| / |G_alt|
        6. g_contrast = highest-KL goal with delta_action=1

    Parameters
    ----------
    policy : GoalConditionedPolicy
        Trained goal-conditioned policy.
    subgoal_vocab : List[SubGoal]
        Sub-goal vocabulary (ordering defines goal indices).
    device : str
        "cpu" or "cuda".
    """

    def __init__(
        self,
        policy: GoalConditionedPolicy,
        subgoal_vocab: List[SubGoal],
        device: str = "cpu",
    ) -> None:
        self.policy = policy.to(device)
        self.policy.eval()
        self.vocab = subgoal_vocab
        self.vocab_size = len(subgoal_vocab)
        self.goal_name_to_idx: Dict[str, int] = {
            sg.name: i for i, sg in enumerate(subgoal_vocab)
        }
        self.device = device

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a single (H, W, C) observation to (1, C, H, W) tensor."""
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if t.shape[-1] in (1, 3):
            t = t.permute(0, 3, 1, 2)
        return t.to(self.device)

    @torch.no_grad()
    def analyze(
        self,
        obs: np.ndarray,
        current_goal_idx: int,
        action_taken: int,
    ) -> Tuple[str, Dict[str, float], Dict[str, int], float]:
        """
        Counterfactual analysis for a single observation / goal / action.

        Parameters
        ----------
        obs : np.ndarray  shape (H, W, C)
        current_goal_idx : int
        action_taken : int

        Returns
        -------
        explanation_text : str
        kl_scores        : Dict[str, float]  goal_name -> KL
        alt_actions      : Dict[str, int]    goal_name -> argmax action
        goal_necessity   : float
        """
        obs_t = self._obs_to_tensor(obs)

        # Current goal distribution
        curr_goal_t = torch.tensor([current_goal_idx], dtype=torch.long, device=self.device)
        logits_curr, _ = self.policy(obs_t, curr_goal_t)
        probs_curr = F.softmax(logits_curr, dim=-1).squeeze(0)   # (n_actions,)
        log_probs_curr = F.log_softmax(logits_curr, dim=-1).squeeze(0)

        kl_scores: Dict[str, float] = {}
        alt_actions: Dict[str, int] = {}
        delta_actions: Dict[str, int] = {}

        for alt_idx, alt_sg in enumerate(self.vocab):
            if alt_idx == current_goal_idx:
                continue
            alt_goal_t = torch.tensor([alt_idx], dtype=torch.long, device=self.device)
            logits_alt, _ = self.policy(obs_t, alt_goal_t)
            probs_alt = F.softmax(logits_alt, dim=-1).squeeze(0)

            # KL(pi_current || pi_alt)
            kl = F.kl_div(
                F.log_softmax(logits_alt, dim=-1).squeeze(0),
                probs_curr,
                reduction="sum",
            ).item()
            kl_scores[alt_sg.name] = max(0.0, kl)

            best_alt_action = int(probs_alt.argmax().item())
            alt_actions[alt_sg.name] = best_alt_action
            delta_actions[alt_sg.name] = int(best_alt_action != action_taken)

        # Goal Necessity
        n_delta = sum(delta_actions.values())
        n_alt = len(delta_actions)
        goal_necessity = n_delta / n_alt if n_alt > 0 else 0.0

        # Contrastive goal: highest KL with delta_action=1
        contrastive_goal: Optional[str] = None
        candidates = {
            name: kl for name, kl in kl_scores.items()
            if delta_actions.get(name, 0) == 1
        }
        if candidates:
            contrastive_goal = max(candidates, key=lambda k: candidates[k])

        # Natural language explanation
        current_name = self.vocab[current_goal_idx].name if current_goal_idx < self.vocab_size else "?"
        explanation_text = self._build_explanation(
            current_name, action_taken, goal_necessity,
            contrastive_goal, kl_scores, delta_actions
        )

        return explanation_text, kl_scores, alt_actions, goal_necessity

    def _build_explanation(
        self,
        current_goal_name: str,
        action_taken: int,
        goal_necessity: float,
        contrastive_goal: Optional[str],
        kl_scores: Dict[str, float],
        delta_actions: Dict[str, int],
    ) -> str:
        """Construct a human-readable explanation string."""
        action_names = ["LEFT", "RIGHT", "FORWARD", "PICKUP", "DROP", "TOGGLE", "DONE"]
        act_name = action_names[action_taken] if action_taken < len(action_names) else str(action_taken)

        lines = [
            f"Goal: {current_goal_name}",
            f"Action taken: {act_name}",
            f"Goal Necessity (GN): {goal_necessity:.2f}",
        ]
        if contrastive_goal:
            kl_val = kl_scores.get(contrastive_goal, 0.0)
            lines.append(
                f"Contrastive goal: {contrastive_goal} "
                f"(KL={kl_val:.3f}, would change action)"
            )
        else:
            lines.append("No alternative goal would change the action.")

        # List top-3 highest KL goals
        top_kl = sorted(kl_scores.items(), key=lambda x: -x[1])[:3]
        if top_kl:
            lines.append("Top-3 divergent goals:")
            for name, kl in top_kl:
                da = delta_actions.get(name, 0)
                lines.append(f"  {name}: KL={kl:.3f}, delta_action={da}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Trajectory-level analysis
    # ------------------------------------------------------------------

    def analyze_trajectory(
        self,
        trajectory: Dict[str, Any],
        goal_assignments: List[Optional[SubGoal]],
    ) -> List[TimestepAnalysis]:
        """
        Run counterfactual analysis at each timestep of a trajectory.

        Parameters
        ----------
        trajectory : dict
            Must contain obs_seq, action_seq.
        goal_assignments : List[Optional[SubGoal]]
            From SubGoalImputation.impute().

        Returns
        -------
        List[TimestepAnalysis]  – one entry per timestep.
        """
        obs_seq: List[np.ndarray] = trajectory["obs_seq"]
        action_seq: List[int] = trajectory["action_seq"]
        T = min(len(obs_seq), len(action_seq), len(goal_assignments))

        results: List[TimestepAnalysis] = []

        for t in range(T):
            sg = goal_assignments[t]
            if sg is None or sg.name not in self.goal_name_to_idx:
                # Use goal 0 as fallback
                current_goal_idx = 0
                sg_obj = self.vocab[0] if self.vocab else SubGoal(name="UNKNOWN", category="")
            else:
                current_goal_idx = self.goal_name_to_idx[sg.name]
                sg_obj = sg

            explanation, kl_scores, alt_actions, gn = self.analyze(
                obs_seq[t], current_goal_idx, action_seq[t]
            )

            # delta_actions reconstruction (needed for contrastive)
            delta_actions: Dict[str, int] = {
                name: int(act != action_seq[t])
                for name, act in alt_actions.items()
            }
            candidates = {
                name: kl for name, kl in kl_scores.items()
                if delta_actions.get(name, 0) == 1
            }
            contrastive = max(candidates, key=lambda k: candidates[k]) if candidates else None

            results.append(TimestepAnalysis(
                t=t,
                current_goal=sg_obj,
                action_taken=action_seq[t],
                kl_scores=kl_scores,
                alt_actions=alt_actions,
                delta_actions=delta_actions,
                goal_necessity=gn,
                contrastive_goal=contrastive,
                explanation_text=explanation,
            ))

        return results

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def visualize_sensitivity_heatmap(
        self,
        trajectory_results: List[TimestepAnalysis],
        save_path: str,
    ) -> None:
        """
        T x |G| heatmap of KL divergence scores.

        Rows = timesteps, Columns = alternative goals.
        Cell colour = KL(pi_current || pi_g').

        Parameters
        ----------
        trajectory_results : List[TimestepAnalysis]
        save_path : str
        """
        if not trajectory_results:
            print("[visualize_sensitivity_heatmap] Empty results; skipping.")
            return

        T = len(trajectory_results)
        # Collect all goal names (preserving vocab order)
        all_goals = [sg.name for sg in self.vocab]
        G = len(all_goals)

        kl_matrix = np.zeros((T, G), dtype=np.float32)
        for t_idx, result in enumerate(trajectory_results):
            for g_idx, gname in enumerate(all_goals):
                kl_matrix[t_idx, g_idx] = result.kl_scores.get(gname, 0.0)

        fig, ax = plt.subplots(figsize=(max(8, G * 0.6), max(5, T * 0.2)))
        sns.heatmap(
            kl_matrix,
            ax=ax,
            cmap="YlOrRd",
            xticklabels=all_goals,
            yticklabels=[str(r.t) for r in trajectory_results],
            cbar_kws={"label": "KL Divergence"},
            linewidths=0.0,
        )
        ax.set_xlabel("Alternative Goal", fontsize=10)
        ax.set_ylabel("Timestep", fontsize=10)
        ax.set_title("Goal Sensitivity Heatmap (KL scores)", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize_sensitivity_heatmap] Saved -> {save_path}")

    def visualize_goal_necessity_timeline(
        self,
        trajectory_results: List[TimestepAnalysis],
        save_path: str,
    ) -> None:
        """
        Plot Goal Necessity (GN) over the trajectory timesteps.

        Parameters
        ----------
        trajectory_results : List[TimestepAnalysis]
        save_path : str
        """
        if not trajectory_results:
            print("[visualize_goal_necessity_timeline] Empty results; skipping.")
            return

        timesteps = [r.t for r in trajectory_results]
        gn_values = [r.goal_necessity for r in trajectory_results]

        # Segment coloring by current goal category
        palette = {
            "NAVIGATE_TO": "#4C9BE8",
            "PICKUP":      "#F4A259",
            "OPEN":        "#6ABF69",
            "REACH_GOAL":  "#E06C75",
            "EXPLORE":     "#C678DD",
            "UNKNOWN":     "#888888",
        }
        colors = [
            palette.get(r.current_goal.category, "#888888")
            for r in trajectory_results
        ]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5),
                                        gridspec_kw={"height_ratios": [3, 1]})

        # -- Main GN plot ------------------------------------------------
        ax1.plot(timesteps, gn_values, color="#333333", linewidth=1.0, zorder=2)
        ax1.fill_between(timesteps, gn_values, alpha=0.2, color="#333333")
        ax1.scatter(timesteps, gn_values, c=colors, s=20, zorder=3, alpha=0.85)
        ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="GN=0.5")
        ax1.set_ylim(-0.05, 1.10)
        ax1.set_xlim(min(timesteps), max(timesteps))
        ax1.set_ylabel("Goal Necessity (GN)", fontsize=10)
        ax1.set_title("Goal Necessity over Time", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=8)

        # Legend for colors
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(facecolor=col, label=cat)
            for cat, col in sorted(palette.items())
            if cat != "UNKNOWN"
        ]
        ax1.legend(handles=handles, loc="upper right", fontsize=7, ncol=3,
                   title="Current goal category")

        # -- Contrastive goal annotation --------------------------------
        for r in trajectory_results:
            if r.contrastive_goal and r.goal_necessity > 0.3:
                ax1.annotate(
                    r.contrastive_goal.split("(")[0],  # short label
                    xy=(r.t, r.goal_necessity),
                    xytext=(r.t, r.goal_necessity + 0.08),
                    fontsize=5,
                    color="#444444",
                    ha="center",
                )

        # -- Action timeline bar chart ----------------------------------
        action_names = ["L", "R", "F", "PK", "DR", "TG", "DN"]
        ax2.bar(
            timesteps,
            [1] * len(timesteps),
            color=[palette.get(r.current_goal.category, "#888888") for r in trajectory_results],
            width=1.0,
            align="center",
        )
        for r in trajectory_results:
            a_label = action_names[r.action_taken] if r.action_taken < len(action_names) else "?"
            ax2.text(r.t, 0.5, a_label, ha="center", va="center", fontsize=5, color="white")
        ax2.set_xlim(min(timesteps) - 0.5, max(timesteps) + 0.5)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax2.set_xlabel("Timestep", fontsize=10)
        ax2.set_ylabel("Action", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize_goal_necessity_timeline] Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Factory: build and train from scratch
# ---------------------------------------------------------------------------

def build_and_train(
    env_name: str,
    subgoal_vocab: List[SubGoal],
    trajectories: List[Dict[str, Any]],
    trajectory_segments: List[List[Tuple[int, int, "SubGoal"]]],
    save_path: Optional[str] = None,
    n_goals: Optional[int] = None,
    n_actions: int = 7,
    obs_shape: Tuple[int, int, int] = (8, 8, 3),
    goal_emb_dim: int = 32,
    total_steps: int = 1_000_000,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cpu",
    negative_reward: float = -0.1,
) -> Tuple[GoalConditionedPolicy, CounterfactualGoalAnalysis]:
    """
    Build a GoalConditionedPolicy, fill a HER buffer from trajectories, train,
    and return the policy + analysis object.

    Parameters
    ----------
    env_name : str
    subgoal_vocab : List[SubGoal]
    trajectories : List[dict]
    trajectory_segments : List of per-trajectory segment lists
    save_path : str, optional  – path to save the trained policy weights
    n_goals : int, optional    – defaults to len(subgoal_vocab)
    n_actions : int
    obs_shape : (H, W, C)
    goal_emb_dim : int
    total_steps : int
    batch_size : int
    lr : float
    device : str
    negative_reward : float

    Returns
    -------
    policy : GoalConditionedPolicy
    analyzer : CounterfactualGoalAnalysis
    """
    n_goals = n_goals or len(subgoal_vocab)
    obs_h, obs_w, obs_c = obs_shape

    policy = GoalConditionedPolicy(
        n_goals=n_goals,
        n_actions=n_actions,
        n_input_channels=obs_c,
        obs_h=obs_h,
        obs_w=obs_w,
        goal_emb_dim=goal_emb_dim,
    )

    # Build HER buffer
    buffer = HERBuffer(subgoal_vocab, negative_reward=negative_reward)
    for traj, segs in zip(trajectories, trajectory_segments):
        buffer.add_trajectory(traj, segs)

    print(f"[build_and_train] HER buffer size: {len(buffer)} experiences")

    if len(buffer) > 0:
        train_goal_conditioned_policy(
            policy=policy,
            her_buffer=buffer,
            total_steps=total_steps,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
    else:
        print("[build_and_train] Warning: HER buffer empty; policy is untrained.")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), save_path)
        print(f"[build_and_train] Policy saved -> {save_path}")

    analyzer = CounterfactualGoalAnalysis(policy, subgoal_vocab, device=device)
    return policy, analyzer


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import gymnasium as gym
    from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

    try:
        from methods.teleological.subgoal_imputation import SubGoalImputation
    except ImportError:
        from subgoal_imputation import SubGoalImputation

    env_name = "MiniGrid-DoorKey-8x8-v0"
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    imputer = SubGoalImputation(env_name)
    vocab = imputer.extract_subgoal_vocab(env)
    print(f"Vocabulary size: {len(vocab)}")

    # Collect a short random trajectory
    obs, _ = env.reset()
    obs_h, obs_w, obs_c = env.observation_space.shape

    grid_states, obs_seq, action_seq, reward_seq = [], [], [], []
    for _ in range(30):
        grid_states.append(env.unwrapped.grid.encode())
        obs_seq.append(obs.copy())
        action = env.action_space.sample()
        obs, r, done, trunc, _ = env.step(action)
        action_seq.append(action)
        reward_seq.append(float(r))
        if done or trunc:
            break

    traj = {
        "obs_seq": obs_seq,
        "action_seq": action_seq,
        "reward_seq": reward_seq,
        "grid_state_seq": grid_states,
    }

    ga, segs, effs = imputer.impute(traj)
    print(f"Segments: {len(segs)}")

    policy = GoalConditionedPolicy(
        n_goals=max(1, len(vocab)),
        n_actions=7,
        n_input_channels=obs_c,
        obs_h=obs_h,
        obs_w=obs_w,
    )
    analyzer = CounterfactualGoalAnalysis(policy, vocab)

    if obs_seq:
        text, kl_scores, alt_acts, gn = analyzer.analyze(obs_seq[0], 0, action_seq[0])
        print(f"\nDemo analysis (t=0):\n{text}")

    env.close()
