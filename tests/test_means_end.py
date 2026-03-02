"""
Tests for Means-End Coherence computation.

Verifies:
1. D_l(i,j) values are in [0, 1].
2. Shape of T_l, Delta_l, kappa_l matrices.
3. Determinism.
4. Integration test on a small subset with synthetic data.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.metrics import compute_means_end_coherence


# ---------------------------------------------------------------------------
# Helper: generate synthetic layer data
# ---------------------------------------------------------------------------

def _make_synthetic_layers(
    n_classes: int = 10,
    n_layers: int = 5,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic T_layers, Delta_layers, kappa_layers.

    D_l values are drawn uniformly in [0, 1] and made symmetric.
    Delta_l = D_l - D_{l-1}.
    kappa_l is drawn uniformly in [0, 1].
    """
    rng = np.random.default_rng(seed)

    T_layers = []
    Delta_layers = []
    kappa_layers = []

    prev_D = None
    for l in range(n_layers):
        D = rng.uniform(0.0, 1.0, size=(n_classes, n_classes))
        D = (D + D.T) / 2  # symmetrise
        np.fill_diagonal(D, 0.0)
        # Gradually increase to simulate growing discriminability
        D = np.clip(D * (l + 1) / n_layers, 0.0, 1.0)

        T_layers.append(D)

        if prev_D is None:
            Delta_layers.append(D.copy())
        else:
            Delta_layers.append(D - prev_D)

        kappa_layers.append(rng.uniform(0.0, 1.0, size=(n_classes, n_classes)))
        prev_D = D

    return T_layers, Delta_layers, kappa_layers


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMeansEndDlRange(unittest.TestCase):
    """D_l(i,j) values should lie in [0, 1] when constructed correctly."""

    def test_dl_range(self) -> None:
        T_layers, _, _ = _make_synthetic_layers(n_classes=10, n_layers=5, seed=0)
        for l, D in enumerate(T_layers):
            self.assertTrue(
                np.all(D >= -1e-6),
                f"Layer {l}: D contains values below 0: min={D.min():.4f}",
            )
            self.assertTrue(
                np.all(D <= 1 + 1e-6),
                f"Layer {l}: D contains values above 1: max={D.max():.4f}",
            )


class TestMeansEndShapes(unittest.TestCase):
    """T_l, Delta_l, and kappa_l should all have consistent shapes."""

    def setUp(self) -> None:
        self.n_classes = 15
        self.n_layers = 5
        self.T_layers, self.Delta_layers, self.kappa_layers = _make_synthetic_layers(
            n_classes=self.n_classes, n_layers=self.n_layers, seed=1
        )

    def test_T_layers_shape(self) -> None:
        for l, T in enumerate(self.T_layers):
            self.assertEqual(
                T.shape,
                (self.n_classes, self.n_classes),
                f"T_layers[{l}] has wrong shape: {T.shape}",
            )

    def test_Delta_layers_shape(self) -> None:
        for l, D in enumerate(self.Delta_layers):
            self.assertEqual(D.shape, (self.n_classes, self.n_classes))

    def test_kappa_layers_shape(self) -> None:
        for l, k in enumerate(self.kappa_layers):
            self.assertEqual(k.shape, (self.n_classes, self.n_classes))

    def test_layer_counts_match(self) -> None:
        self.assertEqual(len(self.T_layers), self.n_layers)
        self.assertEqual(len(self.Delta_layers), self.n_layers)
        self.assertEqual(len(self.kappa_layers), self.n_layers)


class TestMeansEndCoherenceOutput(unittest.TestCase):
    """compute_means_end_coherence should return values in expected ranges."""

    def setUp(self) -> None:
        self.T_layers, self.Delta_layers, self.kappa_layers = _make_synthetic_layers(
            n_classes=10, n_layers=5, seed=2
        )

    def test_monotonicity_in_range(self) -> None:
        result = compute_means_end_coherence(
            self.T_layers, self.Delta_layers, self.kappa_layers
        )
        mono = result["monotonicity"]
        self.assertGreaterEqual(mono, 0.0)
        self.assertLessEqual(mono, 1.0)

    def test_coherence_in_range(self) -> None:
        result = compute_means_end_coherence(
            self.T_layers, self.Delta_layers, self.kappa_layers
        )
        coh = result["coherence"]
        self.assertGreaterEqual(coh, 0.0)
        self.assertLessEqual(coh, 1.0)

    def test_consumption_consistency_in_range(self) -> None:
        result = compute_means_end_coherence(
            self.T_layers, self.Delta_layers, self.kappa_layers
        )
        cc = result["consumption_consistency"]
        self.assertGreaterEqual(cc, -1.0 - 1e-6)
        self.assertLessEqual(cc, 1.0 + 1e-6)

    def test_result_keys(self) -> None:
        result = compute_means_end_coherence(
            self.T_layers, self.Delta_layers, self.kappa_layers
        )
        self.assertIn("monotonicity", result)
        self.assertIn("consumption_consistency", result)
        self.assertIn("coherence", result)

    def test_perfect_monotone_data(self) -> None:
        """When each D_l is strictly >= D_{l-1}, monotonicity should be 1.0."""
        n_classes = 8
        T_layers = []
        Delta_layers = []
        kappa_layers = []
        for l in range(5):
            D = np.full((n_classes, n_classes), 0.1 * l)
            np.fill_diagonal(D, 0.0)
            T_layers.append(D)
            Delta_layers.append(np.zeros_like(D) if l == 0 else T_layers[l] - T_layers[l-1])
            kappa_layers.append(np.random.rand(n_classes, n_classes) * 0.1 * (l + 1))

        result = compute_means_end_coherence(T_layers, Delta_layers, kappa_layers)
        self.assertAlmostEqual(result["monotonicity"], 1.0, places=5)

    def test_single_layer(self) -> None:
        """Single layer should not crash and return a coherence value."""
        T_layers = [np.random.rand(5, 5)]
        Delta_layers = [T_layers[0]]
        kappa_layers = [np.random.rand(5, 5)]
        result = compute_means_end_coherence(T_layers, Delta_layers, kappa_layers)
        self.assertIn("coherence", result)


class TestMeansEndDeterminism(unittest.TestCase):
    """compute_means_end_coherence must be deterministic."""

    def test_determinism(self) -> None:
        T_layers, Delta_layers, kappa_layers = _make_synthetic_layers(
            n_classes=12, n_layers=5, seed=7
        )
        result1 = compute_means_end_coherence(T_layers, Delta_layers, kappa_layers)
        result2 = compute_means_end_coherence(T_layers, Delta_layers, kappa_layers)

        for key in result1:
            self.assertAlmostEqual(
                result1[key], result2[key], places=10,
                msg=f"Mismatch on key '{key}': {result1[key]} vs {result2[key]}",
            )


class TestMeansEndIntegration(unittest.TestCase):
    """Integration test with a small synthetic subset."""

    def test_integration_small_subset(self) -> None:
        """Run full pipeline on a 5-class, 3-layer synthetic setup."""
        n_classes = 5
        n_layers = 3

        T_layers, Delta_layers, kappa_layers = _make_synthetic_layers(
            n_classes=n_classes, n_layers=n_layers, seed=123
        )

        result = compute_means_end_coherence(
            T_layers, Delta_layers, kappa_layers, tolerance=0.02
        )

        # All keys present
        for k in ["monotonicity", "consumption_consistency", "coherence"]:
            self.assertIn(k, result)

        # No NaN/Inf
        for k, v in result.items():
            self.assertTrue(np.isfinite(v), f"result['{k}'] = {v} is not finite")

        # Ranges
        self.assertGreaterEqual(result["monotonicity"], 0.0)
        self.assertLessEqual(result["monotonicity"], 1.0)
        self.assertGreaterEqual(result["coherence"], 0.0)
        self.assertLessEqual(result["coherence"], 1.0)


if __name__ == "__main__":
    unittest.main()
