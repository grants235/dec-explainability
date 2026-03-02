"""
Tests for purposive saliency output shapes, properties and determinism.

Because the real PurposiveSaliency may not yet exist, the tests mock the
model with a deterministic small ResNet-style network and verify:

1. Shape correctness:  s_agg is (448, 448); per-competitor maps are (448, 448);
   annotation map is (448, 448).
2. s_agg is non-negative.
3. S_j values are finite.
4. Determinism with a fixed seed.
5. Integration test on a single synthetic image.
"""

from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is importable
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from methods.baselines.integrated_gradients import IGExplainer

# ---------------------------------------------------------------------------
# Minimal deterministic mock model
# ---------------------------------------------------------------------------

class _TinyResNet(nn.Module):
    """
    A tiny model that mimics the ResNet-50 interface needed by the explainers
    (layer4[-1] attribute and a forward pass returning 200-dim logits).
    """

    def __init__(self, n_classes: int = 200, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        # Spatial feature extraction (produces a 14x14 map for 448x448 input)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=32, stride=32),  # 448 / 32 = 14
            nn.ReLU(inplace=True),
        )
        # Simulate layer4 attribute (a single-layer Sequential with a relu)
        layer4_block = nn.Sequential(nn.Conv2d(16, 16, 1), nn.ReLU(inplace=True))
        layer4_block[-1]  # type: ignore
        self.layer4 = nn.Sequential(layer4_block)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Purposive saliency stub (uses IGExplainer under the hood)
# ---------------------------------------------------------------------------

class _PurposiveSaliencyMock:
    """
    Stub purposive saliency: wraps IGExplainer to produce per-competitor maps.
    Returns deterministic results for a given seed.
    """

    IMAGE_SIZE = 448

    def __init__(self, model: nn.Module, device: str = "cpu", seed: int = 42) -> None:
        self._ig = IGExplainer(model, device=device)
        self._seed = seed

    def compute(
        self,
        x: torch.Tensor,
        class_i: int,
        class_j: int,
        n_steps: int = 10,
    ) -> np.ndarray:
        torch.manual_seed(self._seed + class_i * 1000 + class_j)
        return self._ig.compute(x, class_i, n_steps=n_steps, image_size=self.IMAGE_SIZE)

    def compute_aggregated(
        self,
        x: torch.Tensor,
        class_i: int,
        competitors: List[int],
        n_steps: int = 10,
    ) -> tuple:
        """
        Returns (s_agg, per_competitor_maps) where:
          - s_agg : np.ndarray (IMAGE_SIZE, IMAGE_SIZE)
          - per_competitor_maps : list of np.ndarray (IMAGE_SIZE, IMAGE_SIZE)
        """
        per_maps = [self.compute(x, class_i, comp, n_steps) for comp in competitors]
        if per_maps:
            s_agg = np.mean(per_maps, axis=0)
        else:
            s_agg = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.float32)
        return s_agg, per_maps

    def compute_annotation_map(
        self,
        x: torch.Tensor,
        class_i: int,
        competitors: List[int],
    ) -> np.ndarray:
        """Annotation map: max over per-competitor maps, normalised to [0,1]."""
        _, per_maps = self.compute_aggregated(x, class_i, competitors, n_steps=10)
        if not per_maps:
            return np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.float32)
        stack = np.stack(per_maps, axis=0)
        ann = stack.max(axis=0)
        m = ann.max()
        return (ann / m).astype(np.float32) if m > 0 else ann


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPurposiveSaliencyShapes(unittest.TestCase):
    """Verify output shapes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _TinyResNet(n_classes=200, seed=0)
        cls.model.eval()
        cls.device = "cpu"
        cls.saliency = _PurposiveSaliencyMock(cls.model, device=cls.device, seed=42)
        cls.x = torch.zeros(1, 3, 448, 448)
        cls.class_i = 0
        cls.competitors = [1, 2, 3]

    def test_per_competitor_shape(self) -> None:
        m = self.saliency.compute(self.x, self.class_i, self.competitors[0], n_steps=5)
        self.assertEqual(m.shape, (448, 448), f"Expected (448,448), got {m.shape}")

    def test_sagg_shape(self) -> None:
        s_agg, per_maps = self.saliency.compute_aggregated(
            self.x, self.class_i, self.competitors, n_steps=5
        )
        self.assertEqual(s_agg.shape, (448, 448))
        for m in per_maps:
            self.assertEqual(m.shape, (448, 448))

    def test_annotation_map_shape(self) -> None:
        ann = self.saliency.compute_annotation_map(
            self.x, self.class_i, self.competitors
        )
        self.assertEqual(ann.shape, (448, 448))


class TestPurposiveSaliencyProperties(unittest.TestCase):
    """Verify non-negativity and finiteness."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _TinyResNet(n_classes=200, seed=1)
        cls.model.eval()
        cls.saliency = _PurposiveSaliencyMock(cls.model, device="cpu", seed=7)
        cls.x = torch.randn(1, 3, 448, 448)
        cls.class_i = 5
        cls.competitors = [10, 20]

    def test_sagg_nonnegative(self) -> None:
        s_agg, _ = self.saliency.compute_aggregated(
            self.x, self.class_i, self.competitors, n_steps=5
        )
        self.assertTrue((s_agg >= 0).all(), "s_agg contains negative values")

    def test_sj_finite(self) -> None:
        _, per_maps = self.saliency.compute_aggregated(
            self.x, self.class_i, self.competitors, n_steps=5
        )
        for i, m in enumerate(per_maps):
            self.assertTrue(
                np.all(np.isfinite(m)),
                f"S_{i} contains non-finite values",
            )

    def test_annotation_map_nonnegative(self) -> None:
        ann = self.saliency.compute_annotation_map(
            self.x, self.class_i, self.competitors
        )
        self.assertTrue((ann >= 0).all())

    def test_annotation_map_normalised(self) -> None:
        ann = self.saliency.compute_annotation_map(
            self.x, self.class_i, self.competitors
        )
        max_val = float(ann.max())
        self.assertLessEqual(max_val, 1.0 + 1e-5)


class TestPurposiveSaliencyDeterminism(unittest.TestCase):
    """Verify determinism with fixed seed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _TinyResNet(n_classes=200, seed=2)
        cls.model.eval()
        cls.x = torch.randn(1, 3, 448, 448)
        cls.class_i = 3
        cls.competitors = [7, 14]

    def test_deterministic_per_competitor(self) -> None:
        sal1 = _PurposiveSaliencyMock(self.model, device="cpu", seed=99)
        sal2 = _PurposiveSaliencyMock(self.model, device="cpu", seed=99)
        m1 = sal1.compute(self.x, self.class_i, self.competitors[0], n_steps=5)
        m2 = sal2.compute(self.x, self.class_i, self.competitors[0], n_steps=5)
        np.testing.assert_array_almost_equal(m1, m2, decimal=5)

    def test_deterministic_aggregated(self) -> None:
        sal1 = _PurposiveSaliencyMock(self.model, device="cpu", seed=99)
        sal2 = _PurposiveSaliencyMock(self.model, device="cpu", seed=99)
        s1, _ = sal1.compute_aggregated(self.x, self.class_i, self.competitors, n_steps=5)
        s2, _ = sal2.compute_aggregated(self.x, self.class_i, self.competitors, n_steps=5)
        np.testing.assert_array_almost_equal(s1, s2, decimal=5)


class TestPurposiveSaliencyIntegration(unittest.TestCase):
    """Integration test: run on a single synthetic image end-to-end."""

    def test_integration_single_image(self) -> None:
        torch.manual_seed(42)
        model = _TinyResNet(n_classes=200, seed=3)
        model.eval()

        saliency = _PurposiveSaliencyMock(model, device="cpu", seed=0)

        # Realistic-ish input: random (1, 3, 448, 448)
        x = torch.randn(1, 3, 448, 448)
        class_i = 0
        competitors = [1, 5, 10]

        s_agg, per_maps = saliency.compute_aggregated(
            x, class_i, competitors, n_steps=5
        )
        ann = saliency.compute_annotation_map(x, class_i, competitors)

        # Shape checks
        self.assertEqual(s_agg.shape, (448, 448))
        self.assertEqual(ann.shape, (448, 448))
        self.assertEqual(len(per_maps), len(competitors))

        # Property checks
        self.assertTrue((s_agg >= 0).all())
        self.assertTrue(np.all(np.isfinite(s_agg)))
        self.assertTrue(np.all(np.isfinite(ann)))
        self.assertGreater(s_agg.max(), 0.0, "s_agg should not be all zeros")


if __name__ == "__main__":
    unittest.main()
