"""Utility functions related to training networks.

Functions:
    * evaluate_classifier(model, test_loader, ...): Evaluate a
        classification model.
    * get_transforms(transform_params): Get transforms for a
        classification task.
    * set_device(): Set the device to use for training.
    * set_seeds(repr_params): Set random seeds for reproducibility.
"""


__all__ = [
    "evaluate_classifier",
    "get_transforms",
    "set_device",
    "set_seeds"
]

import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torchmetrics import MetricCollection
from tqdm import tqdm

from src.config import ReproducibilityConf, TransformConf
from src.utils.classification_presets import ClassificationPresets


def evaluate_classifier(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        metrics: MetricCollection,
        device: torch.device
) -> Dict[str, float]:
    """Evaluate a classification model.

    Args:
        model: The classification model to evaluate.
        test_loader: The dataloader providing test samples.
        criterion: The criterion to use for evaluation.
        metrics: The metrics to evaluate the model with.
        device: The device to perform evaluation on.

    Returns:
        The loss along with the computed metrics, evaluated on the test
        set.
    """

    model.eval()
    metrics.reset()
    metrics.to(device)

    running_loss = 0.
    running_samples = 0

    # Set up progress bar
    pbar = tqdm(
        test_loader,
        desc=f"Evaluating {model.__class__.__name__}",
        total=len(test_loader),
        leave=True,
        unit="batch"
    )

    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Make predictions and update metrics
            predictions = model(inputs)
            metrics.update(predictions, targets)

            # Compute loss and accumulate
            loss = criterion(predictions, targets)
            samples = targets.size(dim=0)
            running_loss += loss.item() * samples
            running_samples += samples

    metric_values = metrics.compute()
    results = {
        "Loss": running_loss / running_samples,
        **metric_values
    }

    return results


def get_transforms(
        transform_params: TransformConf
) -> Tuple[ClassificationPresets, ClassificationPresets]:
    """Get transforms for a classification task.

    Args:
        transform_params: The parameters to use for setting up the
          transforms.

    Returns:
        The training and validation transforms for a classification
        task.
    """

    train_transform = ClassificationPresets(
        mean=transform_params.mean,
        std=transform_params.std,
        crop_size=transform_params.crop_size,
        crop_scale=(
            transform_params.crop_scale.lower,
            transform_params.crop_scale.upper
        ),
        crop_ratio=(
            transform_params.crop_ratio.lower,
            transform_params.crop_ratio.upper
        ),
        flip_prob=transform_params.flip_prob,
        is_training=True
    )

    val_transform = ClassificationPresets(
        mean=transform_params.mean,
        std=transform_params.std,
        crop_size=transform_params.crop_size,
        resize_size=transform_params.resize_size,
        is_training=False
    )

    return train_transform, val_transform


def set_device() -> torch.device:
    """Set the device to use for training.

    Returns:
        The device to use for training.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seeds(repr_params: ReproducibilityConf) -> None:
    """Set random seeds for reproducibility.

    Args:
        repr_params: The parameters to use for setting up the random
          seeds.
    """

    random.seed(repr_params.torch_seed)
    np.random.seed(repr_params.torch_seed)
    torch.manual_seed(repr_params.torch_seed)
    torch.cuda.manual_seed_all(repr_params.torch_seed)
    torch.backends.cudnn.deterministic = repr_params.cudnn_deterministic
    torch.backends.cudnn.benchmark = repr_params.cudnn_benchmark
