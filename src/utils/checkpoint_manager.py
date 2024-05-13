"""A class to save model checkpoints in PyTorch."""


import logging
from pathlib import Path
from typing import Optional, List

from omegaconf import DictConfig
import torch
import torch.nn as nn

from src.utils.performance_tracker import PerformanceTracker
from src.utils.training_manager import TrainingManager


class CheckpointManager:
    """A manager to save model checkpoints in PyTorch.

    Params:
        checkpoint_dir: The directory to save model checkpoints in.
        cfg: The training configuration.

    (Additional) Attributes:
        latest_checkpoint: The path pointing to the checkpoint saved
          last, excluding the checkpoint corresponding to the best
          performing model.
        logger: A logger instance to record logs.
    """

    def __init__(
            self,
            checkpoint_dir: str,
            cfg: DictConfig
    ) -> None:
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
        # Set up checkpoint
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
            save_regularly: bool,
            save_best: bool
    ) -> List[str]:
        if not save_regularly and not save_best:
            return ["No checkpoints will be saved during training"]

        if save_regularly:
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
