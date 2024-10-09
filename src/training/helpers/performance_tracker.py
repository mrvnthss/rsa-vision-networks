"""A class to monitor model performance and handle early stopping."""


import logging
from typing import Optional

from numpy import inf, isfinite


class PerformanceTracker:
    """A tracker to monitor model performance and handle early stopping.

    Attributes:
        best_epoch_idx: The epoch index at which the best score was
          observed.
        best_score: The best score observed during training.
        higher_is_better: A flag to indicate whether higher values of
          the metric being tracked reflect better performance.
        is_tracking: A flag to indicate whether performance tracking is
          enabled.
        latest_is_best: A flag to indicate whether the latest score of
          the metric being tracked is the best observed so far.
        logger: The logger instance to record logs.
        min_delta: The minimum change in the performance measure to
          qualify as an improvement.
        patience: The number of consecutive epochs without a new best
          performance to wait before stopping the training process
          early.
        patience_counter: The number of consecutive epochs without a new
          best performance.
        track_for_checkpointing: A flag to indicate whether model
          performance is being tracked for checkpointing.
        track_for_early_stopping: A flag to indicate whether model
          performance is being tracked for early stopping.

    Methods:
        is_patience_exceeded(): Check if the patience is exceeded and
          the training process should be stopped early.
        report_status(): Report the status of performance tracking
          during training.
        update(latest_score, epoch_idx): Update the tracker with the
          latest score.
    """

    def __init__(
            self,
            higher_is_better: bool,
            track_for_checkpointing: bool,
            min_delta: float = 0.0,
            patience: Optional[int] = None
    ) -> None:
        """Initialize the PerformanceTracker instance.

        Args:
            higher_is_better: A flag to indicate whether higher values
              of the metric being tracked reflect better performance.
            track_for_checkpointing: A flag to indicate whether model
              performance should be tracked for checkpointing.
            min_delta: The minimum change in the performance measure to
              qualify as an improvement, i.e., an absolute change of
              less than or equal to ``min_delta`` will not count as a
              new best score.
            patience: The number of consecutive epochs without a new
              best performance to wait before stopping the training
              process early.  Set to None to disable early stopping.

        Raises:
            ValueError: If ``patience`` is neither a positive integer
              nor None.
        """

        self.higher_is_better = higher_is_better
        self.track_for_checkpointing = track_for_checkpointing
        self.min_delta = min_delta

        if patience is None:
            self.patience = inf
        elif patience > 0:
            self.patience = patience
        else:
            raise ValueError(
                f"'patience' should be either a positive integer or None, but got {patience}."
            )

        self.track_for_early_stopping = self.patience < inf
        self.is_tracking = self.track_for_checkpointing or self.track_for_early_stopping

        self.best_epoch_idx = -inf
        self.best_score = -inf if self.higher_is_better else inf
        self.latest_is_best = False
        self.patience_counter = 0

        self.logger = logging.getLogger(__name__)

    def is_patience_exceeded(self) -> bool:
        """Check if the patience is exceeded.

        Returns:
            A boolean indicating whether ``patience`` is exceeded.
        """

        return self.patience_counter >= self.patience

    def report_status(self) -> None:
        """Report the status of performance tracking during training."""

        if not self.is_tracking:
            self.logger.info(
                "Performance tracking is disabled. Model performance will not be monitored for "
                "checkpointing or early stopping."
            )
            return

        best_score = f"{self.best_score:.3f}" if isfinite(self.best_score) else self.best_score
        min_delta = f"{self.min_delta:.3f}"

        if self.track_for_early_stopping:
            patience_counter = f"{self.patience_counter}/{self.patience}"
            if self.track_for_checkpointing:
                self.logger.info(
                    "Performance tracking is enabled for both checkpointing and early stopping. "
                    "Current best score is %s, patience counter is at %s, improvement threshold "
                    "is set to %s.",
                    best_score,
                    patience_counter,
                    min_delta
                )
            else:
                self.logger.info(
                    "Performance tracking is enabled for early stopping only. "
                    "Current best score is %s, patience counter is at %s, improvement threshold "
                    "is set to %s.",
                    best_score,
                    patience_counter,
                    min_delta
                )
        else:
            self.logger.info(
                "Performance tracking is enabled for checkpointing only. "
                "Current best score is %s, improvement threshold is set to %s.",
                best_score,
                min_delta
            )

    def update(
            self,
            latest_score: float,
            epoch_idx: int
    ) -> None:
        """Update the tracker with the latest score.

        Note:
            This method may change the values of ``best_score``,
            ``latest_is_best``, ``best_epoch_idx``, and
            ``patience_counter``.

        Args:
            latest_score: The latest score (of the metric determining
              performance) observed while training the model.
            epoch_idx: The epoch index at which the ``latest_score`` was
              observed.
        """

        if self.higher_is_better:
            self.latest_is_best = latest_score > self.best_score + self.min_delta
        else:
            self.latest_is_best = latest_score < self.best_score - self.min_delta

        if self.latest_is_best:
            self.best_score = latest_score
            self.best_epoch_idx = epoch_idx
            self.patience_counter = 0
        else:
            self.patience_counter += 1
