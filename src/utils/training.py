"""Utility functions related to training networks.

Functions:
    * evaluate_classifier(model, test_loader, ...): Evaluate a
        classification model.
    * get_collate_fn(transform_train_params, num_classes): Get a collate
        function incorporating MixUp & CutMix transforms.
    * get_lr_scheduler(cfg, optimizer): Get the learning rate scheduler
        to use during training.
    * get_train_transform(transform_train_params): Get the transform for
        an image classification task (training).
    * get_val_transform(transform_val_params): Get the transform for an
        image classification task (validation).
    * set_device(): Set the device to use for training.
    * set_seeds(repr_params): Set random seeds for reproducibility.
"""


__all__ = [
    "evaluate_classifier",
    "get_collate_fn",
    "get_lr_scheduler",
    "get_train_transform",
    "get_val_transform",
    "set_device",
    "set_seeds"
]

import random
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms.v2 as T
from hydra.utils import instantiate
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import default_collate
from torchmetrics import MetricCollection
from tqdm import tqdm

from src.config import ReproducibilityConf, TrainClassifierConf, TrainSimilarityConf, \
    TransformTrainConf, TransformValConf, CropScaleConf, CropRatioConf
from src.schedulers.sequential_lr import SequentialLR
from src.utils.classification_transforms import *


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


def get_collate_fn(
        transform_train_params: TransformTrainConf,
        num_classes: int
) -> Optional[Callable]:
    """Get a collate function incorporating MixUp & CutMix transforms.

    Args:
        transform_train_params: The parameters of the transform to use
          during training, specifying the MixUp & CutMix parameters.
        num_classes: The number of classes in the dataset.

    Returns:
        The collate function incorporating MixUp & CutMix transforms (if
        specified).
    """

    mixup_alpha = _get_transform_parameter(transform_train_params, "mixup_alpha", 0.0)
    cutmix_alpha = _get_transform_parameter(transform_train_params, "cutmix_alpha", 0.0)

    mixup_cutmix = _get_mixup_cutmix(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        num_classes=num_classes
    )

    if mixup_cutmix is not None:
        def collate_fn(batch: List) -> Any:
            return mixup_cutmix(*default_collate(batch))
    else:
        collate_fn = default_collate

    return collate_fn


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
        # Populate the ``T_max`` argument of the ``CosineAnnealingLR`` scheduler, if applicable
        if cfg.main_scheduler.name == "CosineAnnealingLR":
            warmup_epochs = _get_warmup_epochs(cfg)
            cfg.main_scheduler.kwargs.T_max = cfg.training.num_epochs - warmup_epochs

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
        transform_train_params: TransformTrainConf
) -> ClassificationTransformsTrain:
    """Get the transform for an image classification task (training).

    Args:
        transform_train_params: The parameters of the transform to use
          during training.

    Returns:
        The transform to use during training.
    """

    # Retrieve parameters from config, automatically handling missing values
    crop_scale = _get_transform_parameter(
        transform_train_params,
        "crop_scale",
        CropScaleConf(lower=1.0, upper=1.0)
    )
    crop_ratio = _get_transform_parameter(
        transform_train_params,
        "crop_ratio",
        CropRatioConf(lower=1.0, upper=1.0)
    )
    flip_prob = _get_transform_parameter(transform_train_params, "flip_prob", 0.0)
    ta_wide = _get_transform_parameter(transform_train_params, "ta_wide", False)
    random_erase_prob = _get_transform_parameter(transform_train_params, "random_erase_prob", 0.0)

    # Initialize transform
    transform = ClassificationTransformsTrain(
        mean=transform_train_params.mean,
        std=transform_train_params.std,
        crop_size=transform_train_params.crop_size,
        crop_scale=crop_scale,
        crop_ratio=crop_ratio,
        flip_prob=flip_prob,
        ta_wide=ta_wide,
        random_erase_prob=random_erase_prob
    )
    return transform


def get_val_transform(
        transform_val_params: TransformValConf
) -> ClassificationTransformsVal:
    """Get the transform for an image classification task (validation).

    Args:
        transform_val_params: The parameters of the transform to use
          during validation.

    Returns:
        The transform to use during validation.
    """

    transform = ClassificationTransformsVal(
        mean=transform_val_params.mean,
        std=transform_val_params.std,
        resize_size=transform_val_params.resize_size,
        crop_size=transform_val_params.crop_size
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


def _get_mixup_cutmix(
        mixup_alpha: float,
        cutmix_alpha: float,
        num_classes: int
) -> Optional[T.RandomChoice]:
    """Get a MixUp/CutMix transform.

    Args:
        mixup_alpha: Value of the two shape parameters of the beta
          distribution used by the ``MixUp`` transform.
        cutmix_alpha: Value of the two shape parameters of the beta
          distribution used by the ``CutMix`` transform.
        num_classes: The number of classes in the dataset.

    Returns:
        A transform that randomly (with equal probability) applies
        either ``MixUp`` or ``CutMix`` with the given parameters (if
        both are specified).
    """

    mixup_cutmix = []

    if mixup_alpha > 0:
        mixup_cutmix.append(
            T.MixUp(
                alpha=mixup_alpha,
                num_classes=num_classes
            )
        )

    if cutmix_alpha > 0:
        mixup_cutmix.append(
            T.CutMix(
                alpha=cutmix_alpha,
                num_classes=num_classes
            )
        )

    if not mixup_cutmix:
        return None
    return T.RandomChoice(mixup_cutmix)


def _get_transform_parameter(
        cfg: TransformTrainConf,
        parameter: str,
        default_value: Any
) -> Any:
    """Retrieve a parameter from the transform configuration.

    Args:
        cfg: The transform configuration.
        parameter: The parameter to retrieve.
        default_value: The default value to return if the parameter is
          not present in the configuration.

    Returns:
        The value of the parameter in the configuration, or the default
        value if the parameter is not present.
    """

    if parameter not in cfg or getattr(cfg, parameter) is None:
        return default_value
    return getattr(cfg, parameter)


def _get_warmup_epochs(
        cfg: Union[TrainClassifierConf, TrainSimilarityConf]
) -> int:
    """Get the number of warmup epochs specified for the training run.

    Args:
        cfg: The training configuration.

    Returns:
        The number of warmup epochs specified for the training run.
    """

    if "warmup_scheduler" not in cfg or cfg.warmup_scheduler is None:
        return 0
    return cfg.warmup_scheduler.warmup_epochs
