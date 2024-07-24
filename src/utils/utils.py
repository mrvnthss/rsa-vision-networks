"""Utility functions used throughout this project.

Functions:
    * evaluate_classifier: Evaluate a classification model on a dataset.
"""


from typing import Dict

import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm


def evaluate_classifier(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> Dict[str, float]:
    """Evaluate a classification model on a dataset.

    Args:
        model: The classification model to evaluate.
        test_loader: The dataloader providing test samples.
        criterion: The criterion to use for evaluation.
        device: The device to perform evaluation on.

    Returns:
        The classification accuracy and the loss evaluated on the test
        set.
    """

    model.eval()

    metric = MulticlassAccuracy(
        num_classes=len(test_loader.dataset.classes),
        top_k=1,
        average="micro",
        multidim_average="global"
    ).to(device)

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
        for features, targets in pbar:
            features, targets = features.to(device), targets.to(device)

            # Make predictions
            predictions = model(features)

            # Track multiclass accuracy
            metric.update(predictions, targets)

            # Compute loss and accumulate
            loss = criterion(predictions, targets)
            samples = targets.size(dim=0)
            running_loss += loss.item() * samples
            running_samples += samples

    results = {
        "mca": metric.compute().item(),
        "loss": running_loss / running_samples
    }

    return results
