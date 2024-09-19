"""Utility functions related to training networks.

Functions:
    * evaluate_classifier(model, test_loader, ...): Evaluate a
        classification model.
    * set_seeds(seed, cudnn_deterministic, cudnn_benchmark): Set random
        seeds for reproducibility.
"""


__all__ = [
    "evaluate_classifier",
    "set_seeds"
]

import random
from typing import Dict

import numpy as np
import torch
from torch import nn
from torchmetrics import MetricCollection
from tqdm import tqdm


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


def set_seeds(
        seed: int,
        cudnn_deterministic: bool,
        cudnn_benchmark: bool
) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: The random seed to use.
        cudnn_deterministic: Whether to enforce deterministic behavior
          of cuDNN.
        cudnn_benchmark: Whether to enable cuDNN benchmark mode.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
