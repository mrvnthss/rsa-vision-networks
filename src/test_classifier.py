"""Evaluate a trained classification model on a dataset.

This script is configured using the Hydra framework, with configuration
details specified in the "src/conf/" directory.  The configuration file
associated with this script is named "test_classifier.yaml".

Typical usage example:

  >>> python test_classifier.py model=lenet dataset=fashionmnist evaluate_on=test
  ...                           model.load_weights_from=<path_to_checkpoint>
"""


from typing import Literal

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from torchmetrics import MetricCollection

from src.base_classes.base_loader import BaseLoader
from src.config import TestClassifierConf
from src.utils.training import evaluate_classifier, get_val_transform, set_device

cs = ConfigStore.instance()
cs.store(name="test_classifier_conf", node=TestClassifierConf)


@hydra.main(version_base=None, config_path="conf", config_name="test_classifier")
def main(cfg: TestClassifierConf) -> None:
    """Evaluate a trained classification model on a dataset."""

    if cfg.evaluate_on in ["train", "val"]:
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
    model = instantiate(cfg.model.kwargs).to(device)
    model.load_state_dict(
        torch.load(
            cfg.model.load_weights_from,
            map_location=device,
            weights_only=False
        )["model_state_dict"]
    )

    # Initialize dataloader providing test samples
    dataset = instantiate(
        cfg.dataset.test_set if cfg.evaluate_on == "test" else cfg.dataset.train_set
    )
    transform = get_val_transform(cfg.dataset.transform_val)
    val_split = cfg.dataloader.val_split if cfg.evaluate_on in ["train", "val"] else None
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
    mode: Literal["main", "val"] = "val" if cfg.evaluate_on == "val" else "main"
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
        "train": "training",
        "val": "validation",
        "test": "test"
    }[cfg.evaluate_on]
    output = [f"Results on {split_str} set:"]
    for metric_name, metric_value in results.items():
        if isinstance(metric_value, torch.Tensor):
            # NOTE: Here, we assume that all metrics return scalar values!
            metric_value = metric_value.item()
        output.append(f"{metric_name}: {metric_value:.3f}")
    print("\n  ".join(output))


if __name__ == "__main__":
    main()
