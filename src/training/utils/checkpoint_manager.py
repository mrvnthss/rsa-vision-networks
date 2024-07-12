"""A class to save model checkpoints in PyTorch."""


import logging
from pathlib import Path
from typing import Optional

import torch
from numpy import inf
from omegaconf import DictConfig
from torch import nn

from src.training.utils.performance_tracker import PerformanceTracker


class CheckpointManager:
    """A manager to save model checkpoints in PyTorch.

    Attributes:
        cfg: The training configuration.
        checkpoint_dir: The directory to save model checkpoints in.
        delete_previous: Whether to delete the previously saved
          checkpoint when saving a new one.
        is_checkpointing: A flag to indicate whether checkpoints are
          saved during training.
        latest_checkpoint: The path pointing to the checkpoint saved
          last, excluding the checkpoint corresponding to the best
          performing model.
        logger: The logger instance to record logs.
        save_best_model: Whether to save the best performing model
          during training.  The checkpoint corresponding to the best
          performing model is saved as "best_performing.pt" and is
          overwritten whenever a new best performing model is found.
        save_frequency: The frequency at which to save checkpoints
          during training.  If None, checkpoints will not be saved
          regularly.

    Methods:
        report_status(): Report the status of checkpoint saving during
          training.
        resume_training(resume_from, model, ...): Resume training from a
          saved checkpoint.
        save_checkpoint(epoch, model, ...): Save a checkpoint.
    """

    def __init__(
            self,
            cfg: DictConfig
    ) -> None:
        """Initialize the CheckpointManager instance.

        Args:
            cfg: The training configuration.

        Raises:
            ValueError: If the save frequency is not a positive integer
              or None.
        """

        self.checkpoint_dir = Path(cfg.checkpoints.dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if cfg.checkpoints.save_frequency is None:
            self.save_frequency = inf
        elif cfg.checkpoints.save_frequency > 0:
            self.save_frequency = cfg.checkpoints.save_frequency
        else:
            raise ValueError("Save frequency must be a positive integer or None.")

        self.save_best_model = cfg.checkpoints.save_best_model
        self.is_checkpointing = self.save_frequency < inf or self.save_best_model

        self.latest_checkpoint: Optional[Path] = None
        self.delete_previous = cfg.checkpoints.delete_previous

        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

    def report_status(self) -> None:
        """Report the status of checkpoint saving during training."""

        if not self.is_checkpointing:
            self.logger.info("No checkpoints are saved during training.")
            return

        is_enabled = [self.save_frequency < inf, self.save_best_model]
        dataset = "training set" if self.cfg.performance.dataset == "Train" else "validation set"

        msgs_enabled = [
            (
                "Regular saving is enabled, checkpoints are saved every "
                f"{'epoch' if self.save_frequency == 1 else f'{self.save_frequency} epochs'}. "
                "Periodically saved checkpoints are "
                f"{'continuously' if self.delete_previous else 'not'} overwritten."
            ),
            f"Best model is saved during training, {self.cfg.performance.metric} on the {dataset} "
            "determines performance."
        ]

        msgs_disabled = [
            "Regular saving is disabled, checkpoints are not saved on a per-epoch basis.",
            "No additional checkpoints based on model performance are saved."
        ]

        for is_enabled, msg_enabled, msg_disabled in zip(is_enabled, msgs_enabled, msgs_disabled):
            self.logger.info(msg_enabled if is_enabled else msg_disabled)

    def resume_training(
            self,
            resume_from: str,
            device: torch.device,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            performance_tracker: PerformanceTracker,
            keep_previous_best_score: bool = True
    ) -> int:
        """Resume training from a saved checkpoint.

        This method loads the model and optimizer states from a saved
        checkpoint and updates the model's and the optimizer's states
        accordingly.  If applicable, relevant parameters of the
        PerformanceTracker instance are updated as well.

        Args:
            resume_from: The path of the checkpoint to resume training
              from.
            device: The device on which to load the checkpoint.  This
              should match the device used during training.
            model: The model to be trained.
            optimizer: The optimizer used for training.
            performance_tracker: The PerformanceTracker instance to
              monitor model performance and handle early stopping.
            keep_previous_best_score: Whether to keep the best score
              found in the checkpoint for early stopping purposes.  If
              set to False, this best score is discarded, and
              performance tracking will start from scratch.

        Returns:
            The epoch index to resume training from.

        Raises:
            ValueError: If the model architecture or the optimizer
              parameters in the training configuration do not match
              those found in the checkpoint.
        """

        # Load checkpoint
        self.logger.info(
            "Training is being resumed. Loading checkpoint %s from %s ...",
            Path(resume_from).name,
            Path(resume_from).parent
        )
        chkpt = torch.load(resume_from, map_location=device)
        self.logger.info("Checkpoint loaded successfully.")

        # Model
        self.logger.info("Loading model state ...")
        try:
            if chkpt["config"].model.name == self.cfg.model.name:
                model.load_state_dict(chkpt["model"])
                self.logger.info("Model state loaded successfully.")
            else:
                raise ValueError(
                    "Model architecture in config does not match architecture found in checkpoint."
                )
        except ValueError as e:
            self.logger.exception("Error occurred while loading model state: %s", e)
            raise

        # Optimizer
        self.logger.info("Loading optimizer state ...")
        try:
            for hparam in chkpt["config"].optimizer:
                if chkpt["config"].optimizer[hparam] != self.cfg.optimizer[hparam]:
                    raise ValueError(
                        "Optimizer in config does not match configuration found in checkpoint."
                    )
            optimizer.load_state_dict(chkpt["optimizer"])
            self.logger.info("Optimizer state loaded successfully.")
        except ValueError as e:
            self.logger.exception("Error occurred while loading optimizer state: %s", e)
            raise

        # PerformanceTracker
        if performance_tracker.is_tracking:
            default_best_score = "-inf" if performance_tracker.higher_is_better else "inf"
            if chkpt["config"].performance.metric != self.cfg.performance.metric:
                # NOTE: The best score is initialized to -inf or inf in the PerformanceTracker
                #       class, so there is no need to set it here.
                self.logger.warning(
                    "Performance metric in config does not match metric found in checkpoint! "
                    "Best score is reset to %s for tracking purposes.",
                    default_best_score
                )
            elif chkpt["config"].performance.dataset != self.cfg.performance.dataset:
                self.logger.warning(
                    "Performance dataset in config does not match dataset found in checkpoint! "
                    "Best score is reset to %s for tracking purposes.",
                    default_best_score
                )
            elif not keep_previous_best_score:
                self.logger.info(
                    "Previous best score is discarded and reset to %s for tracking purposes.",
                    default_best_score
                )
            elif chkpt["best_score"] is not None:
                performance_tracker.best_score = chkpt["best_score"]
                self.logger.info(
                    "Best score (%s/%s) set to %.4f for tracking purposes.",
                    self.cfg.performance.metric,
                    self.cfg.performance.dataset,
                    chkpt["best_score"]
                )
            else:
                self.logger.warning(
                    "No best score found in checkpoint, initialized to %s for tracking purposes.",
                    default_best_score
                )

        return chkpt["epoch_idx"] + 1

    def save_checkpoint(
            self,
            epoch_idx: int,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            best_score: Optional[float] = None,
            is_regular_save: bool = False
    ) -> None:
        """Save a checkpoint.

        Note:
            The training configuration is automatically saved with each
            checkpoint.

        Args:
            epoch_idx: The current epoch index.
            model: The model to save.
            optimizer: The optimizer to save.
            best_score: The best score achieved during training.
            is_regular_save: Whether the checkpoint to save is a
              periodically saved checkpoint.
        """

        # Set up dictionary to save
        state = {
            "epoch_idx": epoch_idx,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": best_score,
            "config": self.cfg
        }
        chkpt_name = f"epoch_{epoch_idx}.pt" if is_regular_save else "best_performing.pt"
        chkpt_path = self.checkpoint_dir / chkpt_name

        # Save checkpoint
        self.logger.debug("Saving checkpoint %s in %s ...", chkpt_name, self.checkpoint_dir)
        torch.save(state, chkpt_path)
        self.logger.debug("Checkpoint saved successfully.")

        # Delete previous checkpoint if necessary
        if is_regular_save:
            if self.delete_previous and self.latest_checkpoint is not None:
                self.logger.debug(
                    "Deleting previous checkpoint %s from %s ...",
                    self.latest_checkpoint.name,
                    self.latest_checkpoint.parent
                )
                Path(self.latest_checkpoint).unlink()
                self.logger.debug("Previous checkpoint deleted successfully.")
            # Update path pointing to latest checkpoint
            self.latest_checkpoint = chkpt_path
