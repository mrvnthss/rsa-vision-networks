"""A class to save model checkpoints in PyTorch."""


import logging
from pathlib import Path
from typing import Optional, List

import torch
from omegaconf import DictConfig
from torch import nn

from src.utils.performance_tracker import PerformanceTracker
from src.utils.training_manager import TrainingManager


class CheckpointManager:
    """A manager to save model checkpoints in PyTorch.

    Attributes:
        cfg: The training configuration.
        checkpoint_dir: The directory to save model checkpoints in.
        latest_checkpoint: The path pointing to the checkpoint saved
            last, excluding the checkpoint corresponding to the best
            performing model.
        logger: A logger instance to record logs.

    Methods:
        get_status(save_periodically, save_best): Get information about
          the status of checkpoint saving.
        resume_training(resume_from, model, ...): Resume training from a
          saved checkpoint.
        save_checkpoint(epoch, model, ...): Save a checkpoint.
    """

    def __init__(
            self,
            checkpoint_dir: str,
            cfg: DictConfig
    ) -> None:
        """Initialize the CheckpointManager instance.

        Args:
            checkpoint_dir: The directory to save model checkpoints in.
            cfg: The training configuration
        """

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg

        self.latest_checkpoint: Optional[Path] = None
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(
            self,
            epoch: int,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            best_score: Optional[float] = None,
            is_best: bool = False,
            delete_previous: bool = False
    ) -> None:
        """Save a checkpoint.

        Args:
            epoch: The epoch number.
            model: The model to save.
            optimizer: The optimizer to save.
            best_score: The best score achieved during training.
            is_best: Whether the checkpoint to save corresponds to the
              best performing model.
            delete_previous: Whether to delete a previously saved
              checkpoint.  Only taken into account if ``is_best`` is
              False.
        """

        # Set up dictionary to save
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": best_score,
            "config": self.cfg
        }
        chkpt_name = "best_performing.pt" if is_best else f"epoch_{epoch}.pt"
        chkpt_path = self.checkpoint_dir / chkpt_name

        # Save checkpoint
        self.logger.debug("Saving checkpoint %s in %s", chkpt_name, self.checkpoint_dir)
        torch.save(state, chkpt_path)
        self.logger.debug("Checkpoint saved successfully")

        if not is_best:
            # Delete previous checkpoint
            if delete_previous and self.latest_checkpoint is not None:
                self.logger.debug(
                    "Deleting previous checkpoint %s from %s",
                    self.latest_checkpoint.name,
                    self.latest_checkpoint.parent
                )
                Path(self.latest_checkpoint).unlink()
                self.logger.debug("Previous checkpoint deleted successfully")
            # Update latest checkpoint
            self.latest_checkpoint = chkpt_path

    def resume_training(
            self,
            resume_from: str,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            train_manager: Optional[TrainingManager] = None,
            performance_tracker: Optional[PerformanceTracker] = None
    ) -> None:
        """Resume training from a saved checkpoint.

        This method loads the model and optimizer states from a saved
        checkpoint and updates the model's and the optimizer's states
        accordingly.  If applicable, all relevant parameters of the
        TrainingManager and PerformanceTracker instances are updated as
        well.

        Args:
            resume_from: The path to the checkpoint to resume training
              from.
            model: The model to be trained.
            optimizer: The optimizer used for training.
            device: The device to train on.
            train_manager: The TrainingManager instance used to train
              the ``model``.
            performance_tracker: The PerformanceTracker instance used
              to train the ``model``.

        Raises:
            ValueError: If the model architecture or the optimizer
              parameters in the configuration do not match those found
              in the checkpoint.
        """

        # Load checkpoint
        self.logger.info("Resuming training from checkpoint")
        self.logger.info(
            "Loading checkpoint %s from %s",
            Path(resume_from).name,
            Path(resume_from).parent
        )
        chkpt = torch.load(resume_from, map_location=device)

        # Load model state
        self.logger.info("Loading model state")
        try:
            if chkpt["config"].model.name == self.cfg.model.name:
                model.load_state_dict(chkpt["model"])
                self.logger.info("Model state loaded successfully")
            else:
                raise ValueError(
                    "Model architecture in config does not match architecture found in checkpoint."
                )
        except ValueError as e:
            self.logger.exception("Error occurred while loading model state: %s", e)
            raise

        # Load optimizer state
        self.logger.info("Loading optimizer state")
        try:
            # Make sure that optimizer parameters match
            for k in chkpt["config"].optimizer:
                if chkpt["config"].optimizer[k] != self.cfg.optimizer[k]:
                    raise ValueError(
                        "Optimizer in config does not match configuration found in checkpoint."
                    )

            optimizer.load_state_dict(chkpt["optimizer"])
            self.logger.info("Optimizer state loaded successfully")
        except ValueError as e:
            self.logger.exception("Error occurred while loading optimizer state: %s", e)
            raise

        # Set epoch in training manager
        if train_manager:
            train_manager.start_epoch = chkpt["epoch"] + 1
            train_manager.epoch = chkpt["epoch"] + 1

        # Update performance tracker's parameters
        if performance_tracker:
            if chkpt["config"].training.performance_metric == self.cfg.training.performance_metric:
                if chkpt["best_score"] is not None:
                    self.logger.info(
                        "Best score (%s) will be set to %.4f for tracking purposes",
                        self.cfg.training.performance_metric,
                        chkpt["best_score"]
                    )
                    performance_tracker.best_score = chkpt["best_score"]
                else:
                    best_score = "-inf" if performance_tracker.higher_is_better else "inf"
                    self.logger.warning(
                        "No best score found in checkpoint, "
                        "will be reset to %s for tracking purposes",
                        best_score
                    )
            else:
                best_score = "-inf" if performance_tracker.higher_is_better else "inf"
                self.logger.warning(
                    "Performance metric in config does not match metric found in checkpoint: "
                    "Best score will be set to %s for tracking purposes",
                    best_score
                )

    def get_status(
            self,
            save_periodically: bool,
            save_best: bool
    ) -> List[str]:
        """Get the status of checkpoint saving during training.

        Args:
            save_periodically: Whether to periodically save model
              checkpoints.
            save_best: Whether to save the best performing model.

        Returns:
            A list of messages describing the status of checkpoint
            saving.
        """

        if not (save_periodically or save_best):
            return ["No checkpoints will be saved during training"]

        if save_periodically:
            msgs = [
                "Regular saving is enabled, checkpoints will be saved every "
                f"{self.cfg.checkpoints.save_frequency} epochs"
            ]
        else:
            msgs = [
                "Regular saving is disabled, checkpoints will not be saved on a per-epoch basis"
            ]

        if save_best:
            msgs.append(
                "The best model will be saved during training; "
                f"{self.cfg.training.performance_metric} determines performance"
            )
        else:
            msgs.append(
                "No additional checkpoints based on performance will be saved"
            )

        return msgs
