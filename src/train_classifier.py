"""Train a classification model in PyTorch.

This script is configured using the Hydra framework, with configuration
details specified in the "src/conf/" directory.  The configuration file
associated with this script is named "train_classifier.yaml".

Typical usage example:

  >>> python train_classifier.py experiment=lenet_fashionmnist/grid_search/batch_size_lr
"""


import logging

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from torchmetrics import MetricCollection

from src.base_classes.base_loader import BaseLoader
from src.config import TrainClassifierConf
from src.training.classification_trainer import ClassificationTrainer
from src.utils.training import get_transforms, set_device, set_seeds

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="train_classifier_conf", node=TrainClassifierConf)


@hydra.main(version_base=None, config_path="conf", config_name="train_classifier")
def main(cfg: TrainClassifierConf) -> None:
    """Train a classification model in PyTorch."""

    # Set target device
    device = set_device()
    logger.info("Target device is set to: %s.", device.type.upper())

    # Set seeds for reproducibility
    set_seeds(cfg.reproducibility)

    # Prepare transforms and dataset
    logger.info("Preparing transforms and dataset ...")
    train_transform, val_transform = get_transforms(cfg.dataset.transform_params)
    dataset = instantiate(cfg.dataset.train_set)

    # Set up dataloaders
    logger.info("Preparing dataloaders ...")
    base_loader = BaseLoader(
        dataset=dataset,
        main_transform=train_transform,
        val_transform=val_transform,
        val_split=cfg.dataloader.val_split,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        split_seed=cfg.reproducibility.split_seed,
        shuffle_seed=cfg.reproducibility.shuffle_seed
    )
    train_loader = base_loader.get_dataloader(mode="main")
    val_loader = base_loader.get_dataloader(mode="val")

    # Set up criterion
    logger.info("Setting up criterion ...")
    criterion = instantiate(cfg.criterion)

    # Instantiate metrics to track during training
    logger.info("Instantiating metrics ...")
    prediction_metrics = MetricCollection({
        name: instantiate(metric) for name, metric in cfg.metrics.items()
    })

    # Instantiate model and optimizer
    logger.info("Instantiating model and optimizer ...")
    model = instantiate(cfg.model.kwargs).to(device)

    optimizer = instantiate(
        cfg.optimizer.kwargs,
        params=model.parameters()
    )

    # Set up learning rate scheduler
    lr_scheduler = None
    if "lr_scheduler" in cfg and cfg.lr_scheduler is not None:
        logger.info("Setting up learning rate scheduler ...")
        lr_scheduler = instantiate(
            cfg.lr_scheduler.kwargs,
            optimizer=optimizer
        )

    # Instantiate trainer and start training
    # NOTE: Training is automatically resumed if a checkpoint is provided.
    logger.info("Setting up trainer ...")
    trainer = ClassificationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        prediction_metrics=prediction_metrics,
        device=device,
        cfg=cfg,
        lr_scheduler=lr_scheduler
    )
    trainer.train()


if __name__ == "__main__":
    main()
