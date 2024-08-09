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
        The classification accuracy (top-1 and top-5) and the loss
        evaluated on the test set.
    """

    model.eval()

    acc_1 = MulticlassAccuracy(
        num_classes=len(test_loader.dataset.classes),
        top_k=1,
        average="micro",
        multidim_average="global"
    ).to(device)

    acc_5 = MulticlassAccuracy(
        num_classes=len(test_loader.dataset.classes),
        top_k=5,
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
            acc_1.update(predictions, targets)
            acc_5.update(predictions, targets)

            # Compute loss and accumulate
            loss = criterion(predictions, targets)
            samples = targets.size(dim=0)
            running_loss += loss.item() * samples
            running_samples += samples

    results = {
        "loss": running_loss / running_samples,
        "acc@1": acc_1.compute().item(),
        "acc@5": acc_5.compute().item()
    }

    return results
