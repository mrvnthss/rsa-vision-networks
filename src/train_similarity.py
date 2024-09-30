"""Train a model using custom representational similarity loss.

This script is configured using the Hydra framework, with configuration
details specified in the "src/conf/" directory.  The configuration file
associated with this script is named "train_similarity.yaml".

Typical usage example:

  >>> python train_similarity.py experiment=lenet_fashionmnist/representational_similarity/weight_transform
"""


import logging

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from torchmetrics import MetricCollection

from src.base_classes.base_loader import BaseLoader
from src.config import TrainSimilarityConf
from src.training.representational_similarity_trainer import RepresentationalSimilarityTrainer
from src.utils.rsa import get_rsa_loss
from src.utils.training import get_collate_fn, get_lr_scheduler, get_train_transform, \
    get_val_transform, set_device, set_seeds

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="train_similarity_conf", node=TrainSimilarityConf)


@hydra.main(version_base=None, config_path="conf", config_name="train_similarity")
def main(cfg: TrainSimilarityConf) -> None:
    """Train a model using custom representational similarity loss."""

    # Set target device
    device = set_device()
    logger.info("Target device is set to: %s.", device.type.upper())

    # Prepare transforms and dataset
    logger.info("Preparing transforms and dataset ...")
    train_transform = get_train_transform(cfg.dataset.transform_train)
    val_transform = get_val_transform(cfg.dataset.transform_val)
    dataset = instantiate(cfg.dataset.train_set)

    # Instantiate metrics to track during training
    logger.info("Instantiating metrics ...")
    prediction_metrics = MetricCollection({
        name: instantiate(metric) for name, metric in cfg.performance.metrics.items()
    })

    # Set up criterion
    logger.info("Setting up criterion ...")
    criterion = get_rsa_loss(repr_similarity_params=cfg.repr_similarity)

    # Set up dataloaders
    logger.info("Preparing dataloaders ...")
    collate_fn = get_collate_fn(cfg.dataset)
    multiprocessing_context = None
    if device.type == "mps" and cfg.dataloader.num_workers > 0:
        multiprocessing_context = "fork"
    base_loader = BaseLoader(
        dataset=dataset,
        main_transform=train_transform,
        val_transform=val_transform,
        val_split=cfg.dataloader.val_split,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        multiprocessing_context=multiprocessing_context,
        seeds=cfg.reproducibility
    )
    train_loader = base_loader.get_dataloader(mode="main")
    val_loader = base_loader.get_dataloader(mode="val")

    # Set seeds for reproducibility
    set_seeds(cfg.reproducibility)

    # Instantiate both models (training and reference) and optimizer
    logger.info("Instantiating models and optimizer ...")
    model_state_dict = torch.load(
        cfg.model.load_weights_from,
        map_location=device,
        weights_only=False
    )["model_state_dict"]

    model_train = instantiate(cfg.model.kwargs).to(device)
    model_train.load_state_dict(model_state_dict)

    model_ref = instantiate(cfg.model.kwargs).to(device)
    model_ref.load_state_dict(model_state_dict)

    optimizer = instantiate(
        cfg.optimizer.kwargs,
        params=model_train.parameters()
    )

    # Set up learning rate scheduler
    logger.info("Setting up learning rate scheduler (if specified) ...")
    lr_scheduler = get_lr_scheduler(
        cfg=cfg,
        optimizer=optimizer
    )

    # Instantiate trainer and start training
    logger.info("Setting up trainer ...")
    # NOTE: Training is automatically resumed if a checkpoint is provided.  It's the user's
    #       responsibility to ensure that the checkpoint is compatible with the current model,
    #       optimizer, and scheduler.
    trainer = RepresentationalSimilarityTrainer(
        model_train=model_train,
        model_ref=model_ref,
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

    # Remove hooks again
    trainer.remove_hooks()


if __name__ == "__main__":
    main()
