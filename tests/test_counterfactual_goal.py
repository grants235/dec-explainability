"""
Tests for counterfactual goal validity and GN/KL metrics.

Verifies:
1. GN score is in [0, 1].
2. KL divergences are non-negative.
3. Output shapes are correct.
4. Determinism.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock
from typing import Any, Dict, List

import numpy as np

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.metrics import compute_counterfactual_validity, compute_gn_entropy_correlation
from methods.utils.trajectory_utils import compute_gn_score


# ---------------------------------------------------------------------------
# Mock goal-conditioned policy
# ---------------------------------------------------------------------------

class _MockGCPolicy:
    """
    Mock GC policy with deterministic, goal-sensitive responses.

    predict(obs, goal) returns (goal % n_actions) to simulate different
    actions for different goals.
    """

    def __init__(self, n_actions: int = 6, seed: int = 42) -> None:
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)

    def predict(self, obs: Any, goal: Any) -> int:
        if isinstance(goal, int):
            return int(goal) % self.n_actions
        return 0

    def get_action_distribution(self, obs: Any, goal: Any) -> np.ndarray:
        rng = np.random.default_rng(hash(repr(goal)) % (2 ** 31))
        dist = rng.dirichlet(np.ones(self.n_actions))
        return dist


def _make_synthetic_trajectories(
    n_traj: int = 5,
    T: int = 30,
    n_actions: int = 6,
    n_subgoals: int = 4,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    trajs = []
    for i in range(n_traj):
        obs    = rng.standard_normal((T, 16)).astype(np.float32)
        acts   = rng.integers(0, n_actions, size=(T,)).astype(np.int32)
        subgoals  = [int(rng.integers(0, n_subgoals)) for _ in range(T)]
        alt_goals = [(s + 1) % n_subgoals for s in subgoals]

        gn_scores  = rng.uniform(0.0, 1.0, size=(T,)).astype(np.float32)
        h_unc = rng.uniform(0.5, np.log(n_actions), size=(T,)).astype(np.float32)
        h_cnd = np.clip(h_unc - rng.uniform(0.0, 0.5, size=(T,)), 0.0, None).astype(np.float32)

        trajs.append({
            "obs":      obs,
            "actions":  acts,
            "subgoals": subgoals,
            "alt_goals": alt_goals,
            "gn_scores":        gn_scores,
            "H_unconditional":  h_unc,
            "H_conditional":    h_cnd,
            "subgoal_indices":  np.array(subgoals, dtype=np.int32),
        })
    return trajs


# ---------------------------------------------------------------------------
# Tests: GN score range
# ---------------------------------------------------------------------------

class TestGNScoreRange(unittest.TestCase):
    """GN score must be in [0, 1]."""

    def test_gn_score_identical(self) -> None:
        dist = np.ones(6) / 6
        gn = compute_gn_score(dist, dist)
        self.assertAlmostEqual(gn, 0.0, places=5)

    def test_gn_score_deterministic_goal(self) -> None:
        p = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
        q = np.ones(6) / 6
        gn = compute_gn_score(p, q)
        self.assertGreaterEqual(gn, 0.0)
        self.assertLessEqual(gn, 1.0 + 1e-6)

    def test_gn_score_random_pairs(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(100):
            p = rng.dirichlet(np.ones(6))
            q = rng.dirichlet(np.ones(6))
            gn = compute_gn_score(p, q)
            self.assertGreaterEqual(gn, -1e-6, f"GN={gn} < 0")
            self.assertLessEqual(gn, 1.0 + 1e-6, f"GN={gn} > 1")

    def test_gn_score_two_actions(self) -> None:
        p = np.array([0.99, 0.01])
        q = np.array([0.5, 0.5])
        gn = compute_gn_score(p, q)
        self.assertGreaterEqual(gn, 0.0)
        self.assertLessEqual(gn, 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Tests: KL divergences
# ---------------------------------------------------------------------------

class TestKLDivergenceNonNegative(unittest.TestCase):
    """KL divergences underlying GN should always be non-negative."""

    def _kl(self, p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        p /= p.sum()
        q /= q.sum()
        return float(np.sum(p * np.log(p / q)))

    def test_kl_nonnegative_random(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(200):
            p = rng.dirichlet(np.ones(6))
            q = rng.dirichlet(np.ones(6))
            kl = self._kl(p, q)
            self.assertGreaterEqual(kl, -1e-10, f"KL = {kl} < 0")

    def test_kl_zero_same_distribution(self) -> None:
        p = np.array([0.2, 0.3, 0.5])
        kl = self._kl(p, p)
        self.assertAlmostEqual(kl, 0.0, places=5)


# ---------------------------------------------------------------------------
# Tests: compute_counterfactual_validity shapes and range
# ---------------------------------------------------------------------------

class TestCounterfactualValidityOutput(unittest.TestCase):
    """Output keys, types, and value ranges for compute_counterfactual_validity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.gc_policy = _MockGCPolicy(n_actions=6, seed=0)
        cls.trajectories = _make_synthetic_trajectories(
            n_traj=5, T=30, n_actions=6, n_subgoals=4, seed=0
        )

    def test_output_keys_present(self) -> None:
        result = compute_counterfactual_validity(
            self.gc_policy, self.trajectories, n_samples=20, horizon=5
        )
        self.assertIn("first_action_agreement", result)
        self.assertIn("mean_trajectory_divergence", result)

    def test_first_action_agreement_range(self) -> None:
        result = compute_counterfactual_validity(
            self.gc_policy, self.trajectories, n_samples=20, horizon=5
        )
        v = result["first_action_agreement"]
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_trajectory_divergence_range(self) -> None:
        result = compute_counterfactual_validity(
            self.gc_policy, self.trajectories, n_samples=20, horizon=5
        )
        v = result["mean_trajectory_divergence"]
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_finite_outputs(self) -> None:
        result = compute_counterfactual_validity(
            self.gc_policy, self.trajectories, n_samples=20, horizon=5
        )
        for k, v in result.items():
            self.assertTrue(np.isfinite(v), f"result['{k}'] = {v} is not finite")


# ---------------------------------------------------------------------------
# Tests: compute_gn_entropy_correlation shapes
# ---------------------------------------------------------------------------

class TestGNEntropyCorrelationShapes(unittest.TestCase):
    """compute_gn_entropy_correlation must return expected output structure."""

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(5)
        cls.step_results = [
            {
                "gn_score":        float(rng.uniform(0, 1)),
                "H_unconditional": float(rng.uniform(0.5, 1.8)),
                "H_conditional":   float(rng.uniform(0.0, 0.9)),
            }
            for _ in range(80)
        ]

    def test_output_keys(self) -> None:
        result = compute_gn_entropy_correlation(self.step_results)
        for key in ["quartile_entropy_reduction", "monotone_quartile",
                    "spearman_rho", "p_value"]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_quartile_list_length(self) -> None:
        result = compute_gn_entropy_correlation(self.step_results)
        qer = result["quartile_entropy_reduction"]
        self.assertEqual(len(qer), 4, f"Expected 4 quartiles, got {len(qer)}")

    def test_spearman_rho_in_range(self) -> None:
        result = compute_gn_entropy_correlation(self.step_results)
        rho = result["spearman_rho"]
        self.assertGreaterEqual(rho, -1.0 - 1e-6)
        self.assertLessEqual(rho, 1.0 + 1e-6)

    def test_p_value_in_range(self) -> None:
        result = compute_gn_entropy_correlation(self.step_results)
        p = result["p_value"]
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_monotone_quartile_is_bool(self) -> None:
        result = compute_gn_entropy_correlation(self.step_results)
        self.assertIsInstance(result["monotone_quartile"], bool)


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    """Both functions must produce identical results when called twice."""

    def test_counterfactual_validity_determinism(self) -> None:
        gc = _MockGCPolicy(n_actions=6, seed=11)
        trajs = _make_synthetic_trajectories(n_traj=3, T=20, seed=11)

        r1 = compute_counterfactual_validity(gc, trajs, n_samples=15, horizon=5)
        r2 = compute_counterfactual_validity(gc, trajs, n_samples=15, horizon=5)

        for key in r1:
            self.assertAlmostEqual(r1[key], r2[key], places=10,
                                   msg=f"Non-determinism in key '{key}'")

    def test_gn_entropy_correlation_determinism(self) -> None:
        rng = np.random.default_rng(17)
        step_results = [
            {
                "gn_score":        float(rng.uniform(0, 1)),
                "H_unconditional": float(rng.uniform(0.5, 1.8)),
                "H_conditional":   float(rng.uniform(0.0, 0.9)),
            }
            for _ in range(60)
        ]

        r1 = compute_gn_entropy_correlation(step_results)
        r2 = compute_gn_entropy_correlation(step_results)

        self.assertEqual(r1["quartile_entropy_reduction"], r2["quartile_entropy_reduction"])
        self.assertAlmostEqual(r1["spearman_rho"], r2["spearman_rho"], places=10)

    def test_gn_score_determinism(self) -> None:
        p = np.array([0.3, 0.2, 0.1, 0.2, 0.1, 0.1])
        q = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.1])
        gn1 = compute_gn_score(p, q)
        gn2 = compute_gn_score(p, q)
        self.assertAlmostEqual(gn1, gn2, places=12)


if __name__ == "__main__":
    unittest.main()
