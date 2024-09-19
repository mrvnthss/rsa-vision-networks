"""Train a model using custom representational similarity loss.

This script is configured using the Hydra framework, with configuration
details specified in the "src/conf/" directory.  The configuration file
associated with this script is named "train_similarity.yaml".

Typical usage example:

  >>> python train_similarity.py experiment=lenet_fashionmnist/representational_similarity/test_run
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
from src.utils.training import get_transforms, set_device, set_seeds

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
    train_transform, val_transform = get_transforms(cfg.dataset.transform_params)
    dataset = instantiate(cfg.dataset.train_set)

    # Instantiate metrics to track during training
    logger.info("Instantiating metrics ...")
    prediction_metrics = MetricCollection({
        name: instantiate(metric) for name, metric in cfg.metrics.items()
    })

    # Set up criterion
    logger.info("Setting up criterion ...")
    criterion = get_rsa_loss(
        compute_name=cfg.rdm.compute.name,
        compute_kwargs=cfg.rdm.compute.kwargs,
        compare_name=cfg.rdm.compare.name,
        compare_kwargs=cfg.rdm.compare.kwargs,
        weight_rsa_score=cfg.repr_similarity.weight_rsa_score,
        rsa_transform_str=cfg.repr_similarity.rsa_transform
    )

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
    train_loader = base_loader.get_dataloader(mode="Main")
    val_loader = base_loader.get_dataloader(mode="Val")

    # Set seeds for reproducibility
    set_seeds(
        seed=cfg.reproducibility.torch_seed,
        cudnn_deterministic=cfg.reproducibility.cudnn_deterministic,
        cudnn_benchmark=cfg.reproducibility.cudnn_benchmark
    )

    # Instantiate both models (training and reference) and optimizer
    logger.info("Instantiating models and optimizer ...")
    model_state_dict = torch.load(
        cfg.model.load_weights_from,
        map_location=device,
        weights_only=False
    )["model_state_dict"]

    model_train = instantiate(cfg.model.architecture).to(device)
    model_train.load_state_dict(model_state_dict)

    model_ref = instantiate(cfg.model.architecture).to(device)
    model_ref.load_state_dict(model_state_dict)

    optimizer = instantiate(
        cfg.optimizer.kwargs,
        params=model_train.parameters()
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
    logger.info("Setting up trainer ...")
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
