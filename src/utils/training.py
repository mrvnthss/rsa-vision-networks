"""Utility functions related to training networks.

Functions:
    * evaluate_classifier(model, test_loader, ...): Evaluate a
        classification model.
    * get_lr_scheduler(cfg, optimizer): Get the learning rate scheduler
        to use during training.
    * get_train_transform(train_transform_params): Get the training
        transforms for an image classification task.
    * get_val_transform(val_transform_params): Get the validation
        transforms for an image classification task.
    * set_device(): Set the device to use for training.
    * set_seeds(repr_params): Set random seeds for reproducibility.
"""


__all__ = [
    "evaluate_classifier",
    "get_lr_scheduler",
    "get_train_transform",
    "get_val_transform",
    "set_device",
    "set_seeds"
]

import random
from typing import Dict, Optional, Union

import numpy as np
import torch
from hydra.utils import instantiate
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MetricCollection
from tqdm import tqdm

from src.config import ReproducibilityConf, TrainClassifierConf, TrainSimilarityConf, \
    TransformTrainConf, TransformValConf
from src.utils.classification_transforms import *
from src.utils.sequential_lr import SequentialLR


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


def get_lr_scheduler(
        cfg: Union[TrainClassifierConf, TrainSimilarityConf],
        optimizer: torch.optim.Optimizer
) -> Optional[LRScheduler]:
    """Get the learning rate scheduler to use during training.

    Args:
        cfg: The training configuration.
        optimizer: The optimizer used during training.
    """

    main_scheduler, warmup_scheduler = None, None

    if "main_scheduler" in cfg and cfg.main_scheduler is not None:
        main_scheduler = instantiate(
            cfg.main_scheduler.kwargs,
            optimizer=optimizer
        )

    if "warmup_scheduler" in cfg and cfg.warmup_scheduler is not None:
        warmup_scheduler = instantiate(
            cfg.warmup_scheduler.kwargs,
            optimizer=optimizer
        )

    if main_scheduler is None:
        # NOTE: ``warmup_scheduler`` can also be None in this case!
        return warmup_scheduler
    if warmup_scheduler is None:
        return main_scheduler

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[cfg.warmup_scheduler.warmup_epochs]
    )

    return lr_scheduler


def get_train_transform(
        train_transform_params: TransformTrainConf
) -> ClassificationTransformsTrain:
    """Get the training transforms for an image classification task.

    Args:
        train_transform_params: The parameters to use for the training
          transforms.

    Returns:
        The training transforms for an image classification task.
    """

    transform = ClassificationTransformsTrain(
        mean=train_transform_params.mean,
        std=train_transform_params.std,
        crop_size=train_transform_params.crop_size,
        crop_scale=train_transform_params.crop_scale,
        crop_ratio=train_transform_params.crop_ratio,
        flip_prob=train_transform_params.flip_prob
    )
    return transform


def get_val_transform(
        val_transform_params: TransformValConf
) -> ClassificationTransformsVal:
    """Get the validation transforms for an image classification task.

    Args:
        val_transform_params: The parameters to use for the validation
          transforms.

    Returns:
        The validation transforms for an image classification task.
    """

    transform = ClassificationTransformsVal(
        mean=val_transform_params.mean,
        std=val_transform_params.std,
        resize_size=val_transform_params.resize_size,
        crop_size=val_transform_params.resize_size
    )
    return transform


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
