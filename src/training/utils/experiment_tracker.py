"""A class to track experiments in PyTorch via TensorBoard."""


import logging
import math
from typing import Dict, List, Optional
from typing_extensions import Literal

import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter


class ExperimentTracker:
    """A tracker to track experiments in PyTorch via TensorBoard.

    Attributes:
        is_tracking: A flag to indicate whether experiment tracking is
          enabled.
        log_indices: The batch indices at which to log to TensorBoard.
        logger: The logger instance to record logs.
        updates_per_epoch: The number of times per epoch to log updates
          to TensorBoard during training and validation.
        writer: The SummaryWriter instance to log updates to
          TensorBoard.

    Methods:
        close(): Close the SummaryWriter instance.
        flush(): Flush the SummaryWriter instance.
        log_scalars(scalars, step, mode): Log scalar values to
          TensorBoard.
    """

    def __init__(
            self,
            cfg: DictConfig,
            updates_per_epoch: Dict[Literal["Train", "Val"], Optional[int]],
            num_train_samples: int,
            num_val_samples: Optional[int] = None
    ) -> None:
        """Initialize the ExperimentTracker instance.

        Args:
            cfg: The training configuration.
            updates_per_epoch: The number of times per epoch to log
              updates to TensorBoard during training and validation.
            num_train_samples: The total number of samples in the
              training dataset.
            num_val_samples: The total number of samples in the
              validation dataset.

        Raises:
            ValueError: If one of the entries in ``updates_per_epoch``
              is neither a positive integer nor None or if
              ``num_val_samples`` is None while
              ``updates_per_epoch['Val']`` is not None.
        """

        # Determine the number of updates per epoch during training and validation
        self.updates_per_epoch = {}
        mode: Literal["Train", "Val"]
        for mode in ["Train", "Val"]:
            if updates_per_epoch[mode] is None or updates_per_epoch[mode] > 0:
                self.updates_per_epoch[mode] = updates_per_epoch[mode]
            else:
                raise ValueError(
                    f"'updates_per_epoch['{mode}']' should be either a positive integer or None, "
                    f"but got {updates_per_epoch[mode]}."
                )

        if self.updates_per_epoch["Val"] is not None and num_val_samples is None:
            raise ValueError(
                "'num_val_samples' must be provided if 'updates_per_epoch['Val']' is not None."
            )

        self.is_tracking = any(updates is not None for updates in self.updates_per_epoch.values())

        self.writer = SummaryWriter(log_dir=cfg.tensorboard.dir)
        self.logger = logging.getLogger(__name__)

        # Set log indices (batch-based, starting at 1)
        self.log_indices: Dict[Literal["Train", "Val"], List[int]] = {}
        if self.is_tracking:
            self._set_log_indices(
                num_samples={
                    "Train": num_train_samples,
                    "Val": num_val_samples
                },
                batch_size=cfg.dataloader.batch_size
            )

    def close(self) -> None:
        """Close the SummaryWriter instance."""

        self.writer.close()

    def flush(self) -> None:
        """Flush the SummaryWriter instance."""

        self.writer.flush()

    def log_scalars(
            self,
            scalars: Dict[str, float],
            step: int,
            mode: Literal["Train", "Val"]
    ) -> None:
        """Log scalar values to TensorBoard.

        Args:
            scalars: A dictionary of scalar values to log.
            step: The current step index (batch-based).
            mode: The mode of the model at the time ``scalars`` were
              collected.  This should be either "Train" or "Val".

        Raises:
            ValueError: If ``mode`` is neither "Train" nor "Val".
        """

        if mode not in ["Train", "Val"]:
            raise ValueError(f"'mode' should be either 'Train' or 'Val', but got {mode}.")

        for key, value in scalars.items():
            self.writer.add_scalar(
                f"{key}/{mode}", value, global_step=step
            )

    def report_status(self) -> None:
        """Report the status of experiment tracking."""

        if not self.is_tracking:
            self.logger.info(
                "Experiment tracking is disabled. No updates will be logged to TensorBoard."
            )
        else:
            msg_enabled = "Experiment tracking is enabled."
            mode: Literal["Train", "Val"]
            for mode in ["Train", "Val"]:
                updates = self.updates_per_epoch[mode]
                mode_str = "training" if mode == "Train" else "validation"
                if updates is None:
                    msg_enabled += f" No updates will be logged to TensorBoard during {mode_str}."
                elif updates == 1:
                    msg_enabled += (
                        " Updates will be logged to TensorBoard at the end of each epoch during "
                        f"{mode_str}."
                    )
                else:
                    msg_enabled += (
                        f" Updates will be logged to TensorBoard {updates} times per epoch during "
                        f"{mode_str}."
                    )
            self.logger.info(msg_enabled)

    def _set_log_indices(
            self,
            num_samples: Dict[Literal["Train", "Val"], Optional[int]],
            batch_size: int
    ) -> None:
        """Set batch indices at which to log to TensorBoard.

        Note:
            The batch indices start from 1 and end at the total number
            of batches.  If the number of batches is less than the
            desired number of updates per epoch, updates are logged to
            TensorBoard at the end of each batch.

        Args:
            num_samples: A dictionary of the total number of samples in
              the training and validation datasets.
            batch_size: The number of samples in each batch.
        """

        mode: Literal["Train", "Val"]
        for mode in ["Train", "Val"]:
            if self.updates_per_epoch[mode] is not None:
                num_batches = int(math.ceil(num_samples[mode] / batch_size))
                if num_batches < self.updates_per_epoch[mode]:
                    self.logger.warning(
                        "The number of mini-batches in the %s set is less than the desired number "
                        "of updates per epoch. Results are logged to TensorBoard after each "
                        "mini-batch. Consider reducing the number of updates.",
                        "training" if mode == "Train" else "validation"
                    )
                    self.updates_per_epoch[mode] = num_batches
                    self.log_indices[mode] = list(range(1, num_batches + 1))
                else:
                    # Linearly spaced sample-based indices
                    sample_indices = torch.linspace(
                        0, num_samples[mode], self.updates_per_epoch[mode] + 1
                    )

                    # Convert to batch-based indices
                    self.log_indices[mode] = (
                        torch.ceil(sample_indices / batch_size)
                    ).int().tolist()[1:]
            else:
                self.log_indices[mode] = []
