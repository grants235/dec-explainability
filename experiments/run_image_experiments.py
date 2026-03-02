"""
Main script for the CUB-200-2011 image explainability experiment pipeline.

Usage
-----
    python experiments/run_image_experiments.py \\
        --config configs/image_config.yaml \\
        --data-dir data/ \\
        --checkpoint models/image/checkpoint/best.pth \\
        --results-dir results/image/

Pipeline steps
--------------
1. Load YAML config
2. Load ResNet-50 checkpoint
3. Load (or recompute) confusion cache
4. Build CUB dataset / test DataLoader
5. Initialise all explainability methods
6. Run evaluation (PBPA, Deletion/Insertion AUC, PS, MEC)
7. Save results (CSV + LaTeX) and qualitative figures
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

# ---------------------------------------------------------------------------
# Resolve project root so that sibling packages are importable regardless
# of the working directory.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from methods.baselines.integrated_gradients import IGExplainer
from methods.baselines.gradcam import GradCAMExplainer
from methods.baselines.shap_explainer import SHAPExplainer
from evaluation.image_eval import run_image_evaluation


# ---------------------------------------------------------------------------
# CUB-200-2011 dataset helper
# ---------------------------------------------------------------------------

class CUBDataset(Dataset):
    """
    Minimal CUB-200-2011 dataset that reads from the standard directory
    layout produced by the official CUB download.

    Expected structure under ``root``::

        root/
          images/          # raw JPEGs
          images.txt       # <idx> <relative_path>
          image_class_labels.txt  # <idx> <class_id>
          train_test_split.txt    # <idx> <is_training>

    Parameters
    ----------
    root : str
        Path to the CUB-200-2011 root directory.
    split : str
        'train' or 'test'.
    transform : callable or None
        Torchvision transform pipeline.
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: Optional[Any] = None,
    ) -> None:
        self.root = root
        self.transform = transform

        images_file = os.path.join(root, "images.txt")
        labels_file = os.path.join(root, "image_class_labels.txt")
        split_file = os.path.join(root, "train_test_split.txt")

        if not all(os.path.exists(f) for f in [images_file, labels_file, split_file]):
            raise FileNotFoundError(
                f"CUB annotation files not found under {root}. "
                "Expected: images.txt, image_class_labels.txt, train_test_split.txt"
            )

        is_train_flag = int(split == "train")
        split_map: Dict[int, int] = {}
        with open(split_file) as f:
            for line in f:
                idx, flag = line.strip().split()
                split_map[int(idx)] = int(flag)

        paths: Dict[int, str] = {}
        with open(images_file) as f:
            for line in f:
                idx, rel_path = line.strip().split(maxsplit=1)
                paths[int(idx)] = rel_path

        labels: Dict[int, int] = {}
        with open(labels_file) as f:
            for line in f:
                idx, cls = line.strip().split()
                labels[int(idx)] = int(cls) - 1  # 0-based

        self.samples: List[tuple] = [
            (os.path.join(root, "images", paths[i]), labels[i])
            for i in sorted(paths.keys())
            if split_map.get(i, 1 - is_train_flag) == is_train_flag
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Confusion cache
# ---------------------------------------------------------------------------

def _load_or_compute_confusion_cache(
    cache_path: str,
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[int, List[int]]:
    """
    Load confusion cache from disk or compute from scratch.

    The cache maps class_idx -> list of top-k confused class indices,
    estimated from the model's softmax outputs on the training set.
    """
    if os.path.exists(cache_path):
        logger.info("Loading confusion cache from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info("Computing confusion cache (this may take a few minutes)...")
    cfg = config.get("confusion_set", {})
    tau = float(cfg.get("tau", 0.02))
    k_min = int(cfg.get("k_min", 3))
    k_max = int(cfg.get("k_max", 10))
    n_classes = int(config["model"]["num_classes"])

    # Accumulate soft-probability sums per (true_class, pred_class)
    confusion_sum = np.zeros((n_classes, n_classes), dtype=np.float64)
    class_count = np.zeros(n_classes, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=-1).cpu().numpy()
            for i, lbl in enumerate(labels.numpy()):
                confusion_sum[lbl] += probs[i]
                class_count[lbl] += 1

    # Normalise by class count
    with np.errstate(divide="ignore", invalid="ignore"):
        confusion_mat = np.where(
            class_count[:, None] > 0,
            confusion_sum / class_count[:, None],
            0.0,
        )
    np.fill_diagonal(confusion_mat, 0.0)

    cache: Dict[int, List[int]] = {}
    for cls in range(n_classes):
        row = confusion_mat[cls]
        above_tau = np.where(row > tau)[0].tolist()
        # Sort by confusion probability descending
        above_tau.sort(key=lambda j: row[j], reverse=True)
        k = max(k_min, min(k_max, len(above_tau)))
        cache[cls] = above_tau[:k]

    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    logger.info("Saved confusion cache to %s", cache_path)
    return cache


# ---------------------------------------------------------------------------
# Annotation loaders
# ---------------------------------------------------------------------------

def _load_part_annotations(data_dir: str, n_test: int) -> Dict[int, Dict[str, Optional[tuple]]]:
    """
    Parse CUB-200-2011 part keypoints for the test split.

    Returns
    -------
    dict  image_idx (0-based test index) -> dict of part_name -> (row, col) or None
    """
    parts_file = os.path.join(data_dir, "parts", "part_locs.txt")
    split_file = os.path.join(data_dir, "train_test_split.txt")
    parts_defs_file = os.path.join(data_dir, "parts", "parts.txt")

    if not os.path.exists(parts_file):
        logger.warning("part_locs.txt not found; returning empty part annotations.")
        return {}

    # Build part id -> name map
    part_id_to_name: Dict[int, str] = {}
    if os.path.exists(parts_defs_file):
        with open(parts_defs_file) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    part_id_to_name[int(parts[0])] = parts[1]

    # Build global image index -> test-split index map
    split_map: Dict[int, int] = {}
    test_idx = 0
    with open(split_file) as f:
        for line in f:
            gidx, flag = line.strip().split()
            if int(flag) == 0:  # 0 = test in some CUB releases
                split_map[int(gidx)] = test_idx
                test_idx += 1

    annotations: Dict[int, Dict[str, Optional[tuple]]] = {}

    with open(parts_file) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue
            gimg_idx, part_id, x, y, visible = (
                int(tokens[0]), int(tokens[1]),
                float(tokens[2]), float(tokens[3]), int(tokens[4]),
            )
            if gimg_idx not in split_map:
                continue
            tidx = split_map[gimg_idx]
            pname = part_id_to_name.get(part_id, f"part_{part_id}")
            if tidx not in annotations:
                annotations[tidx] = {}
            annotations[tidx][pname] = (y, x) if visible else None  # (row, col)

    return annotations


def _load_attribute_annotations(data_dir: str) -> np.ndarray:
    """
    Load CUB binary attribute annotations.

    Returns np.ndarray of shape (n_images_test, n_attributes) or empty array.
    """
    attr_file = os.path.join(data_dir, "attributes", "image_attribute_labels.txt")
    if not os.path.exists(attr_file):
        logger.warning("image_attribute_labels.txt not found.")
        return np.empty((0, 312))
    rows: List[List[int]] = []
    current_img: List[int] = []
    last_img_idx = -1
    with open(attr_file) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 3:
                continue
            img_idx, _, is_present = int(tokens[0]), int(tokens[1]), int(tokens[2])
            if img_idx != last_img_idx and current_img:
                rows.append(current_img)
                current_img = []
            current_img.append(is_present)
            last_img_idx = img_idx
    if current_img:
        rows.append(current_img)
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 312), dtype=np.float32)


def _load_class_attribute_matrix(data_dir: str, n_classes: int) -> np.ndarray:
    """
    Load or compute the class-level mean attribute matrix ``(n_classes, n_attrs)``.
    """
    mat_path = os.path.join(data_dir, "attributes", "class_attribute_labels_continuous.txt")
    if os.path.exists(mat_path):
        rows = []
        with open(mat_path) as f:
            for line in f:
                rows.append([float(v) for v in line.strip().split()])
        return np.array(rows, dtype=np.float32)
    logger.warning("class_attribute_labels_continuous.txt not found; using zeros.")
    return np.zeros((n_classes, 312), dtype=np.float32)


def _build_attr_part_labels(data_dir: str) -> Optional[List[str]]:
    """
    Build a list of length 312 mapping each CUB attribute index to a
    canonical part name (matching PART_TO_ATTR values in metrics.py).

    Reads attributes.txt whose lines look like:
        1 has_bill_shape::curved_(up_or_down)
        2 has_wing_color::blue
        ...

    Tries the following paths in order:
        data_dir/attributes.txt
        data_dir/attributes/attributes.txt
    """
    candidates = [
        os.path.join(data_dir, "attributes.txt"),
        os.path.join(data_dir, "attributes", "attributes.txt"),
    ]
    attrs_file = next((p for p in candidates if os.path.exists(p)), None)
    if attrs_file is None:
        logger.warning(
            "attributes.txt not found (tried %s); PBPA will be skipped.",
            " and ".join(candidates),
        )
        return None

    # keyword substring -> canonical part name
    keyword_to_part = [
        ("bill",        "beak"),
        ("beak",        "beak"),
        ("crown",       "crown"),
        ("forehead",    "crown"),
        ("nape",        "crown"),
        ("left_eye",    "left_eye"),
        ("right_eye",   "right_eye"),
        ("eye",         "left_eye"),
        ("throat",      "throat"),
        ("breast",      "breast"),
        ("belly",       "belly"),
        ("back",        "back"),
        ("upperparts",  "back"),
        ("left_wing",   "left_wing"),
        ("right_wing",  "right_wing"),
        ("wing",        "left_wing"),
        ("tail",        "tail"),
        ("left_leg",    "left_leg"),
        ("right_leg",   "right_leg"),
        ("leg",         "left_leg"),
    ]

    labels: List[str] = []
    with open(attrs_file) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                labels.append("back")
                continue
            attr_name = parts[1].lower().replace("::", "_")
            assigned = "back"  # default
            for kw, part in keyword_to_part:
                if kw in attr_name:
                    assigned = part
                    break
            labels.append(assigned)
    return labels


def _load_class_names(data_dir: str) -> List[str]:
    names_file = os.path.join(data_dir, "classes.txt")
    if not os.path.exists(names_file):
        return [str(i) for i in range(200)]
    names: List[str] = []
    with open(names_file) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            names.append(parts[1] if len(parts) == 2 else parts[0])
    return names


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, n_classes: int, device: torch.device) -> nn.Module:
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint from %s", checkpoint_path)
    else:
        logger.warning("Checkpoint not found at %s; using random weights.", checkpoint_path)

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Purposive saliency wrapper
# ---------------------------------------------------------------------------

class _PurposiveSaliencyWrapper:
    """
    Wraps the real PurposiveSaliency so image_eval.py can call
    .compute(x, true_class, confusion_set, weights) directly.

    Falls back to IGExplainer if PurposiveSaliency is unavailable, exposing
    the same interface (returns a trivial single-map result).
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self._device = device
        try:
            from methods.teleological.purposive_saliency import PurposiveSaliency  # type: ignore
            self._impl = PurposiveSaliency(model, device=str(device))
            self._is_real = True
            logger.info("Using real PurposiveSaliency implementation.")
        except ImportError:
            from methods.baselines.integrated_gradients import IGExplainer
            self._impl = IGExplainer(model, device=str(device))
            self._is_real = False
            logger.warning(
                "PurposiveSaliency not found; falling back to IGExplainer for purposive slot."
            )

    def compute(
        self,
        x: torch.Tensor,
        true_class: int,
        confusion_set: List[int],
        weights: Dict[int, float],
        n_steps: int = 50,
    ):
        """
        Returns (per_competitor_maps, s_agg, annotation_map) matching
        the real PurposiveSaliency.compute() interface.
        """
        if self._is_real:
            return self._impl.compute(x, true_class, confusion_set, weights, n_steps=n_steps)

        # Fallback: IG toward true_class, broadcast as a single-competitor result
        ig_map = self._impl.compute(x, true_class)
        if isinstance(ig_map, torch.Tensor):
            ig_map = ig_map.detach().cpu().numpy()
        ig_map = np.abs(ig_map.astype(np.float32))
        # Wrap in the expected return format
        import torch as _torch
        s_agg = _torch.from_numpy(ig_map)
        per_competitor_maps = {j: s_agg.clone() for j in confusion_set}
        annotation_map = _torch.zeros(ig_map.shape, dtype=_torch.long)
        return per_competitor_maps, s_agg, annotation_map


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CUB-200-2011 image explainability experiments."
    )
    parser.add_argument(
        "--config", default="configs/image_config.yaml",
        help="Path to image_config.yaml",
    )
    parser.add_argument(
        "--data-dir", default="data/",
        help="Root of the CUB-200-2011 dataset",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/image/checkpoint/best.pth",
        help="Path to ResNet-50 checkpoint (.pth)",
    )
    parser.add_argument(
        "--results-dir", default="results/image/",
        help="Directory where results are written",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="DataLoader batch size",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader num_workers",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ---- Config -----------------------------------------------------------
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    logger.info("Using device: %s", device)

    img_size = int(config["dataset"]["image_size"])
    mean = config["dataset"]["normalize_mean"]
    std = config["dataset"]["normalize_std"]
    n_classes = int(config["model"]["num_classes"])

    # ---- Transforms -------------------------------------------------------
    transform = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # ---- Datasets ---------------------------------------------------------
    data_dir = args.data_dir
    logger.info("Loading CUB dataset from %s", data_dir)

    try:
        test_dataset = CUBDataset(data_dir, split="test", transform=transform)
        train_dataset = CUBDataset(data_dir, split="train", transform=transform)
    except FileNotFoundError as exc:
        logger.error("Dataset load error: %s", exc)
        sys.exit(1)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Model ------------------------------------------------------------
    model = _load_model(args.checkpoint, n_classes, device)

    # ---- Confusion cache --------------------------------------------------
    cache_path = os.path.join(args.results_dir, "confusion_cache.pkl")
    confusion_cache = _load_or_compute_confusion_cache(
        cache_path, model, train_loader, device, config
    )

    # ---- Explainers -------------------------------------------------------
    logger.info("Initialising explainability methods...")

    purposive_saliency = _PurposiveSaliencyWrapper(model, device)
    ig_explainer = IGExplainer(model, device=str(device))
    gradcam_explainer = GradCAMExplainer(model, device=str(device))

    # Background for SHAP: take first 200 training images
    logger.info("Collecting SHAP background samples...")
    shap_explainer = SHAPExplainer(
        model,
        background_dataset=train_dataset,
        n_background=200,
        device=str(device),
    )

    # Means-end decomposition (optional)
    means_end_decomp = None
    try:
        from methods.teleological.means_end_decomposition import MeansEndDecomposition  # type: ignore
        means_end_decomp = MeansEndDecomposition(model, device=str(device))
        logger.info("MeansEndDecomposition loaded.")
    except ImportError:
        logger.warning("MeansEndDecomposition not found; skipping MEC metric.")

    # ---- Annotations ------------------------------------------------------
    logger.info("Loading CUB annotations...")
    n_test = len(test_dataset)
    part_annotations = _load_part_annotations(data_dir, n_test)
    attribute_annotations = _load_attribute_annotations(data_dir)
    class_attribute_matrix = _load_class_attribute_matrix(data_dir, n_classes)
    class_names = _load_class_names(data_dir)
    attr_part_labels = _build_attr_part_labels(data_dir)
    part_names = [
        "back", "beak", "belly", "breast", "crown", "forehead",
        "left_eye", "left_leg", "left_wing", "nape",
        "right_eye", "right_leg", "right_wing", "tail", "throat",
    ]

    # ---- Run evaluation ---------------------------------------------------
    logger.info("Starting image evaluation...")
    results = run_image_evaluation(
        model=model,
        test_loader=test_loader,
        confusion_cache=confusion_cache,
        purposive_saliency=purposive_saliency,
        ig_explainer=ig_explainer,
        gradcam_explainer=gradcam_explainer,
        shap_explainer=shap_explainer,
        means_end_decomp=means_end_decomp,
        part_annotations=part_annotations,
        attribute_annotations=attribute_annotations,
        class_attribute_matrix=class_attribute_matrix,
        part_names=part_names,
        class_names=class_names,
        config=config,
        device=device,
        results_dir=args.results_dir,
        attr_part_labels=attr_part_labels,
    )

    # ---- Print summary ----------------------------------------------------
    logger.info("=== RESULTS SUMMARY ===")
    for method, metrics in results.items():
        logger.info("  %s:", method)
        for k, v in metrics.items():
            logger.info("    %-40s %.4f", k, v if not (isinstance(v, float) and np.isnan(v)) else -999)

    # Save summary JSON
    summary_json_path = os.path.join(args.results_dir, "image_summary.json")
    def _ser(x: Any) -> Any:
        if isinstance(x, float) and np.isnan(x):
            return None
        if isinstance(x, np.floating):
            return float(x)
        return x

    with open(summary_json_path, "w") as f:
        import json
        json.dump({m: {k: _ser(v) for k, v in mv.items()} for m, mv in results.items()}, f, indent=2)
    logger.info("Saved summary JSON to %s", summary_json_path)


if __name__ == "__main__":
    main()
