"""Evaluate a trained classification model on a dataset.

This script is configured using the Hydra framework, with configuration
details specified in the "src/conf/" directory.  The configuration file
associated with this script is named "test_classifier.yaml".

Typical usage example:

  >>> python test_classifier.py model=lenet dataset=fashionmnist
  ...                           dataloader.which_split=Test
"""


import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from src.base_classes.base_loader import BaseLoader
from src.config import TestClassifierConf
from src.training.utils.checkpoint_manager import CheckpointManager
from src.utils.utils import evaluate_classifier

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
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load model from checkpoint
    model = instantiate(cfg.model.architecture).to(device)
    checkpoint_manager = CheckpointManager(cfg)
    checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path=cfg.paths.checkpoint,
        device=device
    )
    checkpoint_manager.load_model(
        model=model,
        checkpoint=checkpoint
    )

    # Initialize dataloader providing test samples
    dataset = instantiate(
        cfg.dataset.test_set if cfg.dataloader.which_split == "Test" else cfg.dataset.train_set
    )
    val_split = (
        cfg.dataloader.val_split if cfg.dataloader.which_split in ["Train", "Val"] else None
    )
    test_loader = BaseLoader(
        dataset=dataset,
        val_split=val_split,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        split_seed=cfg.dataloader.split_seed
    )
    if cfg.dataloader.which_split == "Val":
        test_loader = test_loader.get_val_loader()

    # Instantiate criterion
    criterion = instantiate(cfg.criterion)

    # Evaluate model performance
    results = evaluate_classifier(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
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
        output.append(f"{metric_name}: {metric_value:.3f}")
    print("\n  ".join(output))


if __name__ == "__main__":
    main()
