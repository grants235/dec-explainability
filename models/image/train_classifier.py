#!/usr/bin/env python3
"""
Train ResNet-50 on CUB-200-2011 for teleological XAI experiments.

Training spec
-------------
- SGD, momentum=0.9, weight_decay=1e-4
- LR 0.001 for the new FC head, 0.0001 for pretrained layers
- Cosine-annealing LR schedule, 60 epochs
- Data aug: random horizontal flip, random crop 448x448 from 512 resize,
            color jitter (brightness=0.2, contrast=0.2)
- Batch size 16, image size 448x448, ImageNet normalisation

Saved artefacts (in --save-dir / models/image/checkpoint/)
-----------------------------------------------------------
- resnet50_cub200_best.pth      (best val-accuracy weights)
- resnet50_cub200_final.pth     (final-epoch weights)
- class_names.pkl               {int -> str}
- part_names.pkl                {int -> str}
- part_locations.pkl            {img_id -> {part_id -> (x, y, visible)}}
- image_attributes.pkl          {img_id -> np.ndarray (312,) binary}
- class_attribute_matrix.npz    (200, 312) float32 continuous means

Usage
-----
python train_classifier.py \\
    --data-dir /path/to/CUB_200_2011 \\
    --save-dir /path/to/teleological-xai \\
    --num-epochs 60 \\
    --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 200
IMAGE_SIZE  = 448
RESIZE_TO   = 512

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ===========================================================================
# Dataset
# ===========================================================================

class Cub2011Dataset(Dataset):
    """
    CUB-200-2011 dataset with full annotation loading.

    Reads
    -----
    images.txt                                     (img_id, img_path)
    image_class_labels.txt                         (img_id, class_id)
    train_test_split.txt                           (img_id, is_train)
    classes.txt                                    (class_id, class_name)
    bounding_boxes.txt                             (img_id, x, y, width, height)
    parts/parts.txt                                (part_id, part_name)
    parts/part_locs.txt                            (img_id, part_id, x, y, visible)
    attributes/class_attribute_labels_continuous.txt  (200 rows x 312 cols)
    attributes/image_attribute_labels.txt          (img_id, attr_id, is_present,
                                                    certainty, time)
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        train: bool = True,
        use_bbox: bool = False,
        load_parts: bool = True,
        load_attributes: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        root_dir : str
            Path to the CUB_200_2011 directory.
        transform : callable or None
        train : bool
            True -> training split; False -> test split.
        use_bbox : bool
            If True, crop to bounding box before applying transforms.
        load_parts : bool
            Whether to load part-location annotations.
        load_attributes : bool
            Whether to load image-level binary attribute vectors.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.use_bbox = use_bbox

        # ----------------------------------------------------------------
        # Core annotation files
        # ----------------------------------------------------------------
        images_path  = os.path.join(root_dir, "images.txt")
        labels_path  = os.path.join(root_dir, "image_class_labels.txt")
        split_path   = os.path.join(root_dir, "train_test_split.txt")
        classes_path = os.path.join(root_dir, "classes.txt")
        bboxes_path  = os.path.join(root_dir, "bounding_boxes.txt")

        # {img_id -> img_path}
        self._images: Dict[int, str] = self._read_id_value(images_path, str)
        # {img_id -> class_id (1-indexed)}
        self._labels: Dict[int, int] = self._read_id_value(labels_path, int)
        # {img_id -> is_train (0 or 1)}
        self._split:  Dict[int, int] = self._read_id_value(split_path,  int)
        # {img_id -> (x, y, w, h)}
        self._bboxes: Dict[int, Tuple[float, float, float, float]] = \
            self._read_bboxes(bboxes_path)

        # {class_id (1-indexed) -> class_name}
        self._raw_class_names: Dict[int, str] = self._read_id_value(classes_path, str)
        # {class_idx (0-indexed) -> short name (after dot)}
        self.class_names: Dict[int, str] = {
            cid - 1: name.split(".")[-1]
            for cid, name in self._raw_class_names.items()
        }

        # ----------------------------------------------------------------
        # Filter to train or test split
        # ----------------------------------------------------------------
        flag = 1 if train else 0
        self.img_ids: List[int] = sorted(
            img_id for img_id, is_tr in self._split.items() if is_tr == flag
        )

        # ----------------------------------------------------------------
        # Parts
        # ----------------------------------------------------------------
        self.part_names: Dict[int, str] = {}
        # {img_id -> {part_id -> (x, y, visible)}}
        self.part_locations: Dict[int, Dict[int, Tuple[float, float, int]]] = {}

        if load_parts:
            parts_txt  = os.path.join(root_dir, "parts", "parts.txt")
            part_locs  = os.path.join(root_dir, "parts", "part_locs.txt")
            if os.path.exists(parts_txt):
                self.part_names = self._read_id_value(parts_txt, str)
            if os.path.exists(part_locs):
                self.part_locations = self._read_part_locs(part_locs)

        # ----------------------------------------------------------------
        # Attributes
        # ----------------------------------------------------------------
        # {img_id -> np.ndarray (312,) binary}
        self.image_attributes: Dict[int, np.ndarray] = {}
        # (200, 312) float32 continuous class-level means
        self.class_attribute_matrix: Optional[np.ndarray] = None

        if load_attributes:
            img_attr_path   = os.path.join(
                root_dir, "attributes", "image_attribute_labels.txt"
            )
            class_attr_path = os.path.join(
                root_dir, "attributes",
                "class_attribute_labels_continuous.txt"
            )
            if os.path.exists(img_attr_path):
                self.image_attributes = self._read_image_attributes(img_attr_path)
            if os.path.exists(class_attr_path):
                self.class_attribute_matrix = self._read_class_attribute_matrix(
                    class_attr_path
                )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_id_value(path: str, val_type) -> dict:
        """Read a two-column whitespace-separated file into {id -> value}."""
        result = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue
                key = int(parts[0])
                result[key] = val_type(parts[1].strip())
        return result

    @staticmethod
    def _read_bboxes(path: str) -> Dict[int, Tuple[float, float, float, float]]:
        """Read bounding_boxes.txt -> {img_id -> (x, y, width, height)}."""
        result = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                img_id = int(tokens[0])
                x, y, w, h = float(tokens[1]), float(tokens[2]), \
                              float(tokens[3]), float(tokens[4])
                result[img_id] = (x, y, w, h)
        return result

    @staticmethod
    def _read_part_locs(
        path: str,
    ) -> Dict[int, Dict[int, Tuple[float, float, int]]]:
        """
        Read parts/part_locs.txt.

        Format: img_id  part_id  x  y  visible
        Returns {img_id -> {part_id -> (x, y, visible)}}
        """
        result: Dict[int, Dict[int, Tuple[float, float, int]]] = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                img_id  = int(tokens[0])
                part_id = int(tokens[1])
                x       = float(tokens[2])
                y       = float(tokens[3])
                visible = int(tokens[4])
                if img_id not in result:
                    result[img_id] = {}
                result[img_id][part_id] = (x, y, visible)
        return result

    @staticmethod
    def _read_image_attributes(path: str) -> Dict[int, np.ndarray]:
        """
        Read attributes/image_attribute_labels.txt.

        Format: img_id  attr_id  is_present  certainty  time
        Returns {img_id -> np.ndarray (312,) binary (is_present values)}.

        Only the *most-certain* annotation per (img_id, attr_id) is kept when
        duplicates exist (certainty 4 > 3 > 2 > 1).  Ties are resolved by
        taking the first occurrence.
        """
        # Accumulate: {img_id -> {attr_id -> (is_present, certainty)}}
        raw: Dict[int, Dict[int, Tuple[int, int]]] = {}

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 4:
                    continue
                img_id     = int(tokens[0])
                attr_id    = int(tokens[1])
                is_present = int(tokens[2])
                certainty  = int(tokens[3])

                if img_id not in raw:
                    raw[img_id] = {}
                existing = raw[img_id].get(attr_id)
                if existing is None or certainty > existing[1]:
                    raw[img_id][attr_id] = (is_present, certainty)

        # Convert to numpy arrays
        result: Dict[int, np.ndarray] = {}
        for img_id, attr_dict in raw.items():
            vec = np.zeros(312, dtype=np.float32)
            for attr_id, (is_present, _) in attr_dict.items():
                if 1 <= attr_id <= 312:
                    vec[attr_id - 1] = float(is_present)
            result[img_id] = vec

        return result

    @staticmethod
    def _read_class_attribute_matrix(path: str) -> np.ndarray:
        """
        Read attributes/class_attribute_labels_continuous.txt.

        200 rows x 312 columns; whitespace or comma separated.
        Returns np.ndarray of shape (200, 312), dtype float32.
        """
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Handle both space and comma separation
                if "," in line:
                    values = [float(v) for v in line.split(",")]
                else:
                    values = [float(v) for v in line.split()]
                rows.append(values)
        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns
        -------
        (image_tensor, class_label_0indexed)
        """
        img_id = self.img_ids[idx]
        rel_path = self._images[img_id]
        img_path = os.path.join(self.root_dir, "images", rel_path)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Optional bounding-box crop
        if self.use_bbox:
            x, y, w, h = self._bboxes[img_id]
            left   = int(x)
            upper  = int(y)
            right  = int(x + w)
            lower  = int(y + h)
            image  = image.crop((left, upper, right, lower))

        if self.transform is not None:
            image = self.transform(image)

        label = self._labels[img_id] - 1  # 0-indexed

        return image, label

    def get_img_id(self, idx: int) -> int:
        """Return the raw CUB img_id for dataset index idx."""
        return self.img_ids[idx]


# ===========================================================================
# Transforms
# ===========================================================================

def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(RESIZE_TO),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(RESIZE_TO),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ===========================================================================
# Model
# ===========================================================================

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Build a ResNet-50 pretrained on ImageNet with the FC head replaced by a
    new linear layer mapping 2048 -> num_classes.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_param_groups(model: nn.Module, lr_backbone: float, lr_head: float):
    """
    Separate FC-head parameters from backbone parameters so each group can
    receive a different learning rate.
    """
    head_params  = list(model.fc.parameters())
    head_ids     = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params,     "lr": lr_head},
    ]


# ===========================================================================
# Training loop
# ===========================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, top1_accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += batch_size

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "val",
) -> Tuple[float, float]:
    """Evaluate model. Returns (avg_loss, top1_accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"[{desc}]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


# ===========================================================================
# Annotation caching helpers
# ===========================================================================

def save_annotation_caches(
    dataset: Cub2011Dataset,
    checkpoint_dir: str,
) -> None:
    """
    Persist class_names, part_names, part_locations, image_attributes, and
    class_attribute_matrix to the checkpoint directory.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- class_names ----
    with open(os.path.join(checkpoint_dir, "class_names.pkl"), "wb") as f:
        pickle.dump(dataset.class_names, f)
    print(f"  Saved class_names ({len(dataset.class_names)} classes)")

    # ---- part_names ----
    if dataset.part_names:
        with open(os.path.join(checkpoint_dir, "part_names.pkl"), "wb") as f:
            pickle.dump(dataset.part_names, f)
        print(f"  Saved part_names ({len(dataset.part_names)} parts)")

    # ---- part_locations ----
    if dataset.part_locations:
        with open(
            os.path.join(checkpoint_dir, "part_locations.pkl"), "wb"
        ) as f:
            pickle.dump(dataset.part_locations, f)
        print(
            f"  Saved part_locations "
            f"({len(dataset.part_locations)} images)"
        )

    # ---- image_attributes ----
    if dataset.image_attributes:
        with open(
            os.path.join(checkpoint_dir, "image_attributes.pkl"), "wb"
        ) as f:
            pickle.dump(dataset.image_attributes, f)
        print(
            f"  Saved image_attributes "
            f"({len(dataset.image_attributes)} images)"
        )

    # ---- class_attribute_matrix ----
    if dataset.class_attribute_matrix is not None:
        np.savez_compressed(
            os.path.join(checkpoint_dir, "class_attribute_matrix.npz"),
            class_attribute_matrix=dataset.class_attribute_matrix,
        )
        print(
            f"  Saved class_attribute_matrix "
            f"(shape {dataset.class_attribute_matrix.shape})"
        )


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 on CUB-200-2011"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to CUB_200_2011 root directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=".",
        help="Project root; checkpoints saved to <save-dir>/models/image/checkpoint/",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=60,
        help="Number of training epochs (default 60)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default 16)",
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=1e-4,
        help="LR for pretrained backbone (default 1e-4)",
    )
    parser.add_argument(
        "--lr-head",
        type=float,
        default=1e-3,
        help="LR for new FC head (default 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="SGD weight decay (default 1e-4)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum (default 0.9)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default 4)",
    )
    parser.add_argument(
        "--use-bbox",
        action="store_true",
        default=False,
        help="Crop to bounding box before transforms",
    )
    parser.add_argument(
        "--no-load-parts",
        action="store_true",
        default=False,
        help="Skip loading part annotations",
    )
    parser.add_argument(
        "--no-load-attributes",
        action="store_true",
        default=False,
        help="Skip loading attribute annotations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ----------------------------------------------------------------
    # Reproducibility
    # ----------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----------------------------------------------------------------
    # Device
    # ----------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ----------------------------------------------------------------
    # Checkpoint directory
    # ----------------------------------------------------------------
    checkpoint_dir = os.path.join(
        args.save_dir, "models", "image", "checkpoint"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # ----------------------------------------------------------------
    # Datasets and DataLoaders
    # ----------------------------------------------------------------
    train_transform = get_train_transform()
    val_transform   = get_val_transform()

    load_parts      = not args.no_load_parts
    load_attributes = not args.no_load_attributes

    print("Loading training dataset ...")
    train_dataset = Cub2011Dataset(
        root_dir=args.data_dir,
        transform=train_transform,
        train=True,
        use_bbox=args.use_bbox,
        load_parts=load_parts,
        load_attributes=load_attributes,
    )

    print("Loading validation (test) dataset ...")
    val_dataset = Cub2011Dataset(
        root_dir=args.data_dir,
        transform=val_transform,
        train=False,
        use_bbox=args.use_bbox,
        load_parts=False,   # Only need annotations once (from train_dataset)
        load_attributes=False,
    )

    print(f"  Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ----------------------------------------------------------------
    # Save annotation caches (done once, before training)
    # ----------------------------------------------------------------
    print("\nSaving annotation caches ...")
    save_annotation_caches(train_dataset, checkpoint_dir)

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    print("\nBuilding ResNet-50 model ...")
    model = build_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    # ----------------------------------------------------------------
    # Optimizer and scheduler
    # ----------------------------------------------------------------
    param_groups = get_param_groups(
        model, lr_backbone=args.lr_backbone, lr_head=args.lr_head
    )
    optimizer = optim.SGD(
        param_groups,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs
    )
    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
    }

    print(f"\nStarting training for {args.num_epochs} epochs ...\n")
    t0 = time.time()

    for epoch in range(1, args.num_epochs + 1):
        # --- train ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # --- validate ---
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, desc="val"
        )

        # --- LR step ---
        scheduler.step()

        # --- logging ---
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head     = optimizer.param_groups[1]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.num_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
            f"lr_bb={lr_backbone:.2e}  lr_fc={lr_head:.2e}  "
            f"elapsed={elapsed/60:.1f}m"
        )

        # --- save best ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(
                checkpoint_dir, "resnet50_cub200_best.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"  -> New best val_acc={val_acc*100:.2f}%  saved to {best_path}")

    # ----------------------------------------------------------------
    # Save final checkpoint
    # ----------------------------------------------------------------
    final_path = os.path.join(checkpoint_dir, "resnet50_cub200_final.pth")
    torch.save(
        {
            "epoch": args.num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "history": history,
        },
        final_path,
    )
    print(f"\nFinal checkpoint saved to {final_path}")

    # ----------------------------------------------------------------
    # Save training history
    # ----------------------------------------------------------------
    history_path = os.path.join(checkpoint_dir, "training_history.npz")
    np.savez_compressed(
        history_path,
        train_loss=np.array(history["train_loss"]),
        train_acc=np.array(history["train_acc"]),
        val_loss=np.array(history["val_loss"]),
        val_acc=np.array(history["val_acc"]),
    )
    print(f"Training history saved to {history_path}")

    total_time = time.time() - t0
    print(
        f"\nTraining complete.  "
        f"Best val_acc={best_val_acc*100:.2f}%  "
        f"Total time={total_time/60:.1f}m"
    )


if __name__ == "__main__":
    main()
