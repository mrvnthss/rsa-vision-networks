"""A class to save model checkpoints in PyTorch."""


from pathlib import Path
from typing import Optional

import torch
from torch import nn


class CheckpointManager:
    """A manager to save model checkpoints in PyTorch.

    Params:
        checkpoint_dir: The directory to save model checkpoints in.

    (Additional) Attributes:
        latest_checkpoint: The path pointing to the checkpoint saved
          last, excluding the checkpoint corresponding to the best
          performing model.
    """

    def __init__(
            self,
            checkpoint_dir: str
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.latest_checkpoint: Optional[Path] = None

    def save_checkpoint(
            self,
            epoch: int,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            best_score: Optional[float] = None,
            performance_metric: Optional[str] = None,
            is_best: bool = False,
            delete_previous: bool = False
    ) -> None:
        # Save model checkpoint
        state = {
            "epoch": epoch,
            "arch": type(model).__name__,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": best_score,
            "performance_metric": performance_metric
        }
        chkpt_name = "best_performing.pt" if is_best else f"epoch_{epoch}.pt"
        chkpt_path = self.checkpoint_dir / chkpt_name
        torch.save(state, chkpt_path)

        # Delete previous checkpoint and update the latest checkpoint only for periodic saves
        if not is_best:
            if delete_previous and self.latest_checkpoint:
                Path(self.latest_checkpoint).unlink()
            self.latest_checkpoint = chkpt_path
