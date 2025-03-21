"""A class to save model checkpoints in PyTorch."""


import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from numpy import inf
from omegaconf import DictConfig
from torch import nn

from src.training.helpers.performance_tracker import PerformanceTracker


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
        load_checkpoint(checkpoint_path, device): Load a saved
          checkpoint.
        report_status(): Report the status of checkpoint saving during
          training.
        resume_training(resume_from, device, ...): Resume training from
          a saved checkpoint.
        save_checkpoint(epoch_idx, model_state_dict, ...): Save a
          checkpoint.
    """

    def __init__(
            self,
            cfg: DictConfig,
            run_id: Optional[int] = None
    ) -> None:
        """Initialize the CheckpointManager instance.

        Note:
            The CheckpointManager instance makes use of the following
            entries of the training configuration ``cfg``:
              * checkpoints.delete_previous
              * checkpoints.save_best_model
              * checkpoints.save_frequency
              * paths.checkpoints
              * performance.evaluate_on
              * performance.evaluation_metric

        Args:
            cfg: The training configuration.
            run_id: Optional run ID to distinguish multiple runs using
              the same configuration.  Used to save checkpoints in
              separate directories.

        Raises:
            ValueError: If ``cfg.checkpoints.save_frequency`` is neither
              a positive integer nor None.
        """

        self.logger = logging.getLogger(__name__)

        is_dir_specified = "checkpoints" in cfg.paths and cfg.paths.checkpoints is not None

        if "checkpoints" in cfg and not is_dir_specified:
            self.logger.warning(
                "The 'checkpoints' key is present in the training configuration, but no "
                "directory is specified to save checkpoints in. Checkpoints will not be saved."
            )

        if "checkpoints" not in cfg or not is_dir_specified:
            # Disable checkpointing altogether
            self.checkpoint_dir = None
            self.save_frequency = inf
            self.save_best_model = False
            self.is_checkpointing = False
            self.delete_previous = False
        else:
            self.checkpoint_dir = Path(cfg.paths.checkpoints)
            if run_id is not None:
                self.checkpoint_dir = self.checkpoint_dir / f"run{run_id}"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            if "save_frequency" not in cfg.checkpoints or cfg.checkpoints.save_frequency is None:
                self.save_frequency = inf
            elif cfg.checkpoints.save_frequency > 0:
                self.save_frequency = cfg.checkpoints.save_frequency
            else:
                raise ValueError(
                    "'save_frequency' should be either a positive integer or None, "
                    f"but got {cfg.checkpoints.save_frequency}."
                )

            self.save_best_model = (
                cfg.checkpoints.save_best_model if "save_best_model" in cfg.checkpoints else False
            )

            self.is_checkpointing = self.save_frequency < inf or self.save_best_model

            self.delete_previous = (
                cfg.checkpoints.delete_previous if "delete_previous" in cfg.checkpoints else False
            )

        self.latest_checkpoint: Optional[Path] = None
        self.cfg = cfg

    def load_checkpoint(
            self,
            checkpoint_path: str,
            device: torch.device
    ) -> Dict[str, Any]:
        """Load a saved checkpoint.

        Args:
            checkpoint_path: The path of the checkpoint to load.
            device: The device on which to load the checkpoint.

        Returns:
            The loaded checkpoint.
        """

        self.logger.info(
            "Loading checkpoint %s from %s ...",
            Path(checkpoint_path).name,
            Path(checkpoint_path).parent
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.logger.info("Checkpoint loaded successfully.")
        return checkpoint

    def report_status(self) -> None:
        """Report the status of checkpoint saving during training."""

        if not self.is_checkpointing:
            self.logger.info("No checkpoints are saved during training.")
            return

        is_enabled = [self.save_frequency < inf, self.save_best_model]
        dataset = (
            "training set" if self.cfg.performance.evaluate_on == "train" else "validation set"
        )

        msgs_enabled = [
            (
                "Regular saving is enabled, checkpoints are saved every "
                f"{'epoch' if self.save_frequency == 1 else f'{self.save_frequency} epochs'}. "
                "Periodically saved checkpoints are "
                f"{'continuously' if self.delete_previous else 'not'} overwritten."
            ),
            f"Best model is saved during training, {self.cfg.performance.evaluation_metric} on "
            f"the {dataset} determines performance."
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
            keep_previous_best_score: bool,
            lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ) -> int:
        """Resume training from a saved checkpoint.

        This method loads the model, optimizer, and learning rate
        scheduler (optional) states from a saved checkpoint and updates
        the model's, optimizer's and scheduler's states accordingly.
        If applicable, relevant parameters of the PerformanceTracker
        instance are updated as well.

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
            lr_scheduler: The scheduler used to adjust the learning rate
              during training.

        Returns:
            The epoch index to resume training from.
        """

        # Load checkpoint and update states
        checkpoint = self.load_checkpoint(resume_from, device)
        if lr_scheduler is not None:
            self.logger.info("Updating model, optimizer, and learning rate scheduler states ...")
        else:
            self.logger.info("Updating model and optimizer states ...")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # Update PerformanceTracker instance, if necessary
        if performance_tracker.is_tracking:
            self._init_performance_tracker(
                performance_tracker=performance_tracker,
                keep_previous_best_score=keep_previous_best_score,
                checkpoint=checkpoint
            )

        return checkpoint["epoch_idx"] + 1

    def save_checkpoint(
            self,
            epoch_idx: int,
            model_state_dict: Dict[str, Any],
            optimizer_state_dict: Dict[str, Any],
            lr_scheduler_state_dict: Optional[Dict[str, Any]] = None,
            best_score: Optional[float] = None,
            best_epoch_idx: Optional[int] = None,
            is_regular_save: bool = False
    ) -> None:
        """Save a checkpoint.

        Note:
            The training configuration is automatically saved with each
            checkpoint.

        Args:
            epoch_idx: The current epoch index.
            model_state_dict: The model state dictionary to save.
            optimizer_state_dict: The optimizer state dictionary to
              save.
            lr_scheduler_state_dict: The scheduler state dictionary to
              save.
            best_score: The best score achieved during training.
            best_epoch_idx: The epoch index at which the best score was
              observed.
            is_regular_save: Whether the checkpoint to save is a
              periodically saved checkpoint.
        """

        # Set up dictionary to save
        state = {
            "epoch_idx": epoch_idx,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "lr_scheduler_state_dict": lr_scheduler_state_dict,
            "best_score": best_score,
            "best_epoch_idx": best_epoch_idx,
            "config": self.cfg
        }
        checkpoint_name = f"epoch_{epoch_idx}.pt" if is_regular_save else "best_performing.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save checkpoint
        self.logger.debug("Saving checkpoint %s in %s ...", checkpoint_name, self.checkpoint_dir)
        torch.save(state, checkpoint_path)
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
            self.latest_checkpoint = checkpoint_path

    def _init_performance_tracker(
            self,
            performance_tracker: PerformanceTracker,
            keep_previous_best_score: bool,
            checkpoint: Dict[str, Any]
    ) -> None:
        """Initialize the PerformanceTracker instance.

        Args:
            performance_tracker: The PerformanceTracker instance to
              initialize.
            keep_previous_best_score: Whether to keep the best score
              found in the checkpoint for early stopping purposes.  If
              set to False, this best score is discarded, and
              performance tracking will start from scratch.
            checkpoint: The checkpoint containing the best score to
              initialize the PerformanceTracker instance with.
        """

        default_best_score = "-inf" if performance_tracker.higher_is_better else "inf"
        checkpoint_evaluation_metric = checkpoint["config"].performance.evaluation_metric
        checkpoint_evaluate_on = checkpoint["config"].performance.evaluate_on
        if checkpoint_evaluation_metric != self.cfg.performance.evaluation_metric:
            # NOTE: The best score is initialized to -inf or inf in the PerformanceTracker
            #       class, so there is no need to reset it here.
            self.logger.warning(
                "Metric in training configuration determining performance does not match metric "
                "found in checkpoint! Best score is reset to %s for tracking purposes.",
                default_best_score
            )
        elif checkpoint_evaluate_on != self.cfg.performance.evaluate_on:
            self.logger.warning(
                "Dataset split in training configuration on which to evaluate performance does "
                "not match split found in checkpoint! Best score is reset to %s for tracking "
                "purposes.",
                default_best_score
            )
        elif not keep_previous_best_score:
            self.logger.info(
                "Previous best score is discarded and reset to %s for tracking purposes.",
                default_best_score
            )
        elif checkpoint["best_score"] is not None:
            # NOTE: We assume that the ``best_epoch_idx`` is always set when the ``best_score``
            #       is set in the checkpoint.
            performance_tracker.best_score = checkpoint["best_score"]
            performance_tracker.best_epoch_idx = checkpoint["best_epoch_idx"]
            self.logger.info(
                "Best score (%s on the %s set) set to %.4f (achieved after %d epochs) for "
                "tracking purposes.",
                self.cfg.performance.evaluation_metric,
                "training" if self.cfg.performance.evaluate_on == "train" else "validation",
                performance_tracker.best_score,
                performance_tracker.best_epoch_idx
            )
        else:
            self.logger.warning(
                "No best score found in checkpoint, initialized to %s for tracking purposes.",
                default_best_score
            )
