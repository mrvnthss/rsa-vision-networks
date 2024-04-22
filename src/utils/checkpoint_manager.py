"""A class to save model checkpoints in PyTorch."""


import logging
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig
import torch
from torch import nn

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
        # Save model checkpoint
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": best_score,
            "config": self.cfg
        }
        chkpt_name = "best_performing.pt" if is_best else f"epoch_{epoch}.pt"
        chkpt_path = self.checkpoint_dir / chkpt_name
        torch.save(state, chkpt_path)

        # Delete previous checkpoint and update the latest checkpoint only for periodic saves
        if not is_best:
            if delete_previous and self.latest_checkpoint:
                Path(self.latest_checkpoint).unlink()
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
        chkpt = torch.load(resume_from, map_location=device)

        # Load model state
        if chkpt["config"].model.name == self.cfg.model.name:
            model.load_state_dict(chkpt["model"])
        else:
            raise ValueError(
                "Model architecture in configuration does not match model "
                "architecture found in checkpoint."
            )

        # Load optimizer state
        if chkpt["config"].optimizer.type == self.cfg.optimizer.type:
            optimizer.load_state_dict(chkpt["optimizer"])
        else:
            raise ValueError(
                "Optimizer type in configuration does not match "
                "optimizer type found in checkpoint."
            )

        # Update training manager's parameters
        if train_manager:
            train_manager.start_epoch = chkpt["epoch"] + 1
            train_manager.epoch = chkpt["epoch"] + 1

        # Update performance tracker's parameters
        if performance_tracker:
            if chkpt["config"].training.performance_metric == self.cfg.training.performance_metric:
                if chkpt["best_score"]:
                    performance_tracker.best_score = chkpt["best_score"]
                else:
                    starting_best_score = "-inf" if performance_tracker.higher_is_better else "inf"
                    self.logger.warning(
                        "No previous best score found in checkpoint, best score will be reset to "
                        "%s for tracking purposes.",
                        starting_best_score
                    )
            else:
                self.logger.warning(
                    "Performance metric in configuration does not match performance metric "
                    "found in checkpoint. Any previous best score will be discarded, tracking "
                    "starts from scratch."
                )
