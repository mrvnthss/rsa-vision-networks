"""Evaluate a trained classification model on a dataset.

This script is configured using the Hydra framework, with configuration
details specified in the "src/conf/" directory.  The configuration file
associated with this script is named "test_classifier.yaml".

Typical usage example:

  >>> python test_classifier.py model=lenet dataset=fashionmnist
  ...                           dataloader.which_split=Test
  ...                           model_checkpoint=<path_to_checkpoint>
"""


from typing import Literal

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from torchmetrics import MetricCollection

from src.base_classes.base_loader import BaseLoader
from src.config import TestClassifierConf
from src.training.helpers.checkpoint_manager import CheckpointManager
from src.utils.classification_presets import ClassificationPresets
from src.utils.training import evaluate_classifier, set_device

cs = ConfigStore.instance()
cs.store(name="test_classifier_conf", node=TestClassifierConf)


@hydra.main(version_base=None, config_path="conf", config_name="test_classifier")
def main(cfg: TestClassifierConf) -> None:
    """Evaluate a trained classification model on a dataset."""

    if cfg.dataloader.which_split in ["Train", "Val"]:
        if (cfg.dataloader.val_split is None
                or cfg.dataloader.val_split <= 0
                or cfg.dataloader.val_split >= 1):
            raise ValueError(
                "The validation split should be either None or a float in the range (0, 1), "
                f"but got {cfg.dataloader.val_split}."
            )

    # Set target device
    device = set_device()

    # Load model from checkpoint
    model = instantiate(cfg.model.architecture).to(device)
    checkpoint_manager = CheckpointManager(cfg)
    checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path=cfg.model_checkpoint,
        device=device
    )
    checkpoint_manager.load_model(
        model=model,
        checkpoint=checkpoint
    )

    # Initialize dataloader providing test samples
    # NOTE: The ``resize_size`` defines the size to which to resize the image to before performing
    #       the center crop.  The ``crop_size`` defines the size of the center crop and should thus
    #       match the input size expected by the network.
    transform = ClassificationPresets(
        mean=cfg.dataset.transform_params.mean,
        std=cfg.dataset.transform_params.std,
        crop_size=cfg.dataset.transform_params.crop_size,
        resize_size=cfg.dataset.transform_params.resize_size,
        is_training=False
    )
    dataset = instantiate(
        cfg.dataset.test_set if cfg.dataloader.which_split == "Test" else cfg.dataset.train_set
    )
    val_split = (
        cfg.dataloader.val_split if cfg.dataloader.which_split in ["Train", "Val"] else None
    )
    test_loader = BaseLoader(
        dataset=dataset,
        main_transform=transform,
        val_transform=transform,
        val_split=val_split,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        split_seed=cfg.reproducibility.split_seed
    )
    mode: Literal["Main", "Val"] = "Val" if cfg.dataloader.which_split == "Val" else "Main"
    test_loader = test_loader.get_dataloader(mode=mode)

    # Instantiate criterion
    criterion = instantiate(cfg.criterion)

    # Instantiate metrics
    metrics = MetricCollection({
        name: instantiate(metric) for name, metric in cfg.metrics.items()
    })

    # Evaluate model performance
    results = evaluate_classifier(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        metrics=metrics,
        device=device
    )

    # Print results to console
    split_str = {
        "Train": "training",
        "Val": "validation",
        "Test": "test"
    }[cfg.dataloader.which_split]
    output = [f"Results on {split_str} set:"]
    for metric_name, metric_value in results.items():
        if isinstance(metric_value, torch.Tensor):
            # NOTE: Here, we assume that all metrics return scalar values!
            metric_value = metric_value.item()
        output.append(f"{metric_name}: {metric_value:.3f}")
    print("\n  ".join(output))


if __name__ == "__main__":
    main()
