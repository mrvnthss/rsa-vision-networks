"""Train a model for image classification in PyTorch.

This script is configured using the Hydra framework, with configuration
details specified in the "src/conf/" directory.  The configuration file
associated with this script is named "train_classifier.yaml".

Typical usage example:

  >>> python train_classifier.py experiment=lenet_fashionmnist/grid_search/batch_size_lr
"""


import logging

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from torchmetrics import MetricCollection

from src.base_classes.base_loader import BaseLoader
from src.config import TrainClassifierConf
from src.training.classification_trainer import ClassificationTrainer
from src.utils.classification_presets import ClassificationPresets

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="train_classifier_conf", node=TrainClassifierConf)


@hydra.main(version_base=None, config_path="conf", config_name="train_classifier")
def main(cfg: TrainClassifierConf) -> None:
    """Train a model for image classification in PyTorch."""

    # Set random seeds for reproducibility
    torch.manual_seed(cfg.seeds.torch)
    torch.cuda.manual_seed_all(cfg.seeds.torch)

    # Set target device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Target device is set to: %s.", device.type.upper())

    # Prepare transforms and dataset
    logger.info("Preparing transforms and dataset ...")
    train_transform = ClassificationPresets(
        mean=cfg.dataset.transform_params.mean,
        std=cfg.dataset.transform_params.std,
        crop_size=cfg.dataset.transform_params.crop_size,
        crop_scale=(
            cfg.dataset.transform_params.crop_scale["lower"],
            cfg.dataset.transform_params.crop_scale["upper"]
        ),
        flip_prob=cfg.dataset.transform_params.flip_prob,
        is_training=True
    )
    val_transform = ClassificationPresets(
        mean=cfg.dataset.transform_params.mean,
        std=cfg.dataset.transform_params.std,
        crop_size=cfg.dataset.transform_params.crop_size,
        resize_size=cfg.dataset.transform_params.resize_size,
        is_training=False
    )
    # NOTE: Transforms are handled by the ``BaseLoader`` class, see below.
    dataset = instantiate(cfg.dataset.train_set)

    if cfg.training.num_folds is None:
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
            split_seed=cfg.seeds.split,
            shuffle_seed=cfg.seeds.shuffle
        )
        train_loader = base_loader.get_dataloader(mode="Main")
        val_loader = base_loader.get_dataloader(mode="Val")

        # Instantiate model, criterion, and optimizer
        logger.info("Instantiating model, setting up criterion and optimizer ...")
        model = instantiate(cfg.model.architecture).to(device)
        criterion = instantiate(cfg.criterion)
        optimizer = instantiate(
            {k: cfg.optimizer[k] for k in cfg.optimizer if k not in ["name", "params"]},
            params=model.parameters()
        )
        cfg.optimizer.params = optimizer.state_dict()["param_groups"]

        # Instantiate metrics to track during training
        logger.info("Instantiating metrics ...")
        metrics = MetricCollection({
            name: instantiate(metric) for name, metric in cfg.metrics.items()
        })

        # Instantiate trainer and start training
        # NOTE: Training is automatically resumed if a checkpoint is provided.
        logger.info("Setting up trainer ...")
        trainer = ClassificationTrainer(
            model,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            metrics,
            device,
            cfg
        )
        trainer.train()
    elif cfg.training.num_folds > 1:
        pass  # TODO: Implement k-fold cross-validation
    else:
        raise ValueError(
            f"'cfg.training.num_folds' should be either None or an integer greater than 1, "
            f"but got {cfg.training.num_folds}."
        )


if __name__ == "__main__":
    main()
