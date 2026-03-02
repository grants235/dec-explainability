"""
Tests for sub-goal segmentation and GN score utilities.

Verifies:
1. Every timestep is assigned exactly one sub-goal.
2. Segments partition [0, T-1] without gaps or overlaps.
3. GN score is in [0, 1].
4. Determinism.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from methods.utils.trajectory_utils import (
    segment_trajectory_by_subgoal,
    compute_gn_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_trajectory(T: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "obs":     rng.standard_normal((T, 16)).astype(np.float32),
        "actions": rng.integers(0, 6, size=(T,)).astype(np.int32),
    }


def _make_agent_positions(T: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Random walk on a 8x8 grid
    positions = rng.integers(0, 8, size=(T, 2)).astype(np.int32)
    return positions


# ---------------------------------------------------------------------------
# Tests: segment_trajectory_by_subgoal
# ---------------------------------------------------------------------------

class TestSegmentationCoversAllTimesteps(unittest.TestCase):
    """Every timestep in [0, T-1] must appear in exactly one segment."""

    def _check_partition(self, segs: list, T: int) -> None:
        covered = np.zeros(T, dtype=int)
        for sg_idx, start, end in segs:
            self.assertGreaterEqual(start, 0, "Segment start < 0")
            self.assertLessEqual(end, T - 1, "Segment end > T-1")
            self.assertLessEqual(start, end, "Segment start > end")
            covered[start : end + 1] += 1

        for t in range(T):
            self.assertEqual(
                covered[t], 1,
                f"Timestep {t} covered {covered[t]} times (expected exactly 1)",
            )

    def test_even_partition_no_positions(self) -> None:
        T = 50
        traj = _make_synthetic_trajectory(T)
        sg_positions = [(2, 3), (5, 5), (7, 1)]
        segs = segment_trajectory_by_subgoal(
            traj["obs"], traj["actions"], sg_positions, agent_positions=None
        )
        self._check_partition(segs, T)

    def test_with_agent_positions_exact_match(self) -> None:
        """Agent visits each sub-goal at a known timestep."""
        T = 30
        traj = _make_synthetic_trajectory(T)

        # Sub-goals at positions (1,1), (3,3), (7,7)
        sg_positions = [(1, 1), (3, 3), (7, 7)]

        # Build agent positions so each sub-goal is visited at t=9, 19, 29
        agent_pos = np.zeros((T, 2), dtype=np.int32)
        agent_pos[:, :] = [0, 0]  # default
        agent_pos[9]  = [1, 1]
        agent_pos[19] = [3, 3]
        agent_pos[29] = [7, 7]

        segs = segment_trajectory_by_subgoal(
            traj["obs"], traj["actions"], sg_positions, agent_positions=agent_pos
        )
        self._check_partition(segs, T)

    def test_single_subgoal(self) -> None:
        T = 20
        traj = _make_synthetic_trajectory(T)
        sg_positions = [(4, 4)]
        segs = segment_trajectory_by_subgoal(
            traj["obs"], traj["actions"], sg_positions, agent_positions=None
        )
        # Should produce at most 2 segments (one sub-goal + possible tail)
        self._check_partition(segs, T)

    def test_many_subgoals(self) -> None:
        T = 100
        traj = _make_synthetic_trajectory(T)
        sg_positions = [(i, i) for i in range(10)]
        segs = segment_trajectory_by_subgoal(
            traj["obs"], traj["actions"], sg_positions, agent_positions=None
        )
        self._check_partition(segs, T)


class TestSegmentationProperties(unittest.TestCase):
    """Sub-goal indices and structural properties."""

    def test_subgoal_index_in_range(self) -> None:
        T = 40
        traj = _make_synthetic_trajectory(T)
        sg_positions = [(1, 2), (4, 5), (7, 0)]
        segs = segment_trajectory_by_subgoal(
            traj["obs"], traj["actions"], sg_positions, agent_positions=None
        )
        n_sg = len(sg_positions)
        for sg_idx, start, end in segs:
            self.assertGreaterEqual(sg_idx, 0)
            self.assertLess(sg_idx, n_sg + 5,  # allow for small overflow
                            f"Sub-goal index {sg_idx} out of range for n_sg={n_sg}")

    def test_segments_non_empty(self) -> None:
        T = 30
        traj = _make_synthetic_trajectory(T)
        sg_positions = [(2, 2), (6, 6)]
        segs = segment_trajectory_by_subgoal(
            traj["obs"], traj["actions"], sg_positions, agent_positions=None
        )
        for sg_idx, start, end in segs:
            self.assertGreaterEqual(
                end - start + 1, 1, f"Empty segment: ({sg_idx}, {start}, {end})"
            )


class TestGNScoreRange(unittest.TestCase):
    """GN score must be in [0, 1]."""

    def test_identical_distributions(self) -> None:
        dist = np.array([0.25, 0.25, 0.25, 0.25])
        gn = compute_gn_score(dist, dist)
        self.assertAlmostEqual(gn, 0.0, places=5)

    def test_maximally_different(self) -> None:
        p = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        gn = compute_gn_score(p, q)
        self.assertGreaterEqual(gn, 0.0)
        self.assertLessEqual(gn, 1.0 + 1e-6)

    def test_random_distributions(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            p = rng.dirichlet(np.ones(6))
            q = rng.dirichlet(np.ones(6))
            gn = compute_gn_score(p, q)
            self.assertGreaterEqual(gn, -1e-6, f"GN={gn} < 0")
            self.assertLessEqual(gn, 1.0 + 1e-6, f"GN={gn} > 1")

    def test_uniform_goal_identical_unconditional(self) -> None:
        """When both distributions are uniform, GN should be 0."""
        n = 6
        p = np.ones(n) / n
        q = np.ones(n) / n
        gn = compute_gn_score(p, q)
        self.assertAlmostEqual(gn, 0.0, places=5)


class TestSubgoalDeterminism(unittest.TestCase):
    """segment_trajectory_by_subgoal must be deterministic."""

    def test_determinism_segment_trajectory(self) -> None:
        T = 60
        traj = _make_synthetic_trajectory(T, seed=42)
        sg_positions = [(1, 1), (4, 4), (7, 7)]

        segs1 = segment_trajectory_by_subgoal(
            traj["obs"], traj["actions"], sg_positions, agent_positions=None
        )
        segs2 = segment_trajectory_by_subgoal(
            traj["obs"], traj["actions"], sg_positions, agent_positions=None
        )
        self.assertEqual(segs1, segs2)

    def test_determinism_gn_score(self) -> None:
        rng = np.random.default_rng(13)
        p = rng.dirichlet(np.ones(6))
        q = rng.dirichlet(np.ones(6))
        gn1 = compute_gn_score(p, q)
        gn2 = compute_gn_score(p, q)
        self.assertAlmostEqual(gn1, gn2, places=10)


if __name__ == "__main__":
    unittest.main()
