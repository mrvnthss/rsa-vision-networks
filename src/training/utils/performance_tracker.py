"""A class to track model performance in PyTorch."""


from typing import Dict

from numpy import inf


class PerformanceTracker:
    """A tracker to monitor model performance in PyTorch.

    Attributes:
        best_score: The best score of the ``performance_metric``
          observed so far.
        higher_is_better: A flag to indicate whether higher values
          of the ``performance_metric`` reflect better performance.
        latest_is_best: A flag to indicate whether the latest score
          of the ``performance_metric`` is the best observed so far.
        metrics: A dictionary of metrics to track.
        patience: The number of consecutive epochs without a new best
          performance to wait before stopping the training process.
          Set to ``inf`` to disable early stopping.
        patience_counter: The number of consecutive epochs without a new
          best performance.
        performance_metric: The metric to evaluate model performance.

    Methods:
        get_status(): Get the status of the tracker.
        is_patience_exceeded(): Check if ``patience`` is exceeded
          and the training process should be stopped early.
        update(metrics): Update the tracker with the latest metrics.
    """

    def __init__(
            self,
            metrics: Dict[str, float],
            performance_metric: str,
            higher_is_better: bool,
            patience: int
    ) -> None:
        """Initialize the PerformanceTracker instance.

        Args:
            metrics: A dictionary of metrics to track.
            performance_metric: The metric to evaluate model
              performance.
            higher_is_better: A flag to indicate whether higher values
              of the ``performance_metric`` reflect better performance.
            patience: The number of consecutive epochs without a new
              best performance to wait before stopping the training
              process.  Set to ``inf`` to disable early stopping.
        """

        self.metrics = metrics
        self.performance_metric = performance_metric
        assert self.performance_metric in self.metrics
        self.higher_is_better = higher_is_better
        self.patience = patience

        self.best_score = -inf if self.higher_is_better else inf
        self.latest_is_best = False
        self.patience_counter = 0

    def update(
            self,
            metrics: Dict[str, float]
    ) -> None:
        """Update the tracker with the latest metrics.

        Args:
            metrics: A dictionary of metrics to track.
        """

        assert self.performance_metric in metrics
        self.metrics = metrics
        latest_score = self.metrics[self.performance_metric]

        if self.higher_is_better:
            self.latest_is_best = latest_score > self.best_score
        else:
            self.latest_is_best = latest_score < self.best_score

        if self.latest_is_best:
            self.best_score = latest_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def is_patience_exceeded(self) -> bool:
        """Check if ``patience`` is exceeded.

        Returns:
            A boolean indicating whether ``patience`` is exceeded.
        """

        return self.patience_counter >= self.patience

    def get_status(self) -> str:
        """Get the status of the tracker.

        Returns:
            A message describing the status of the tracker.
        """

        if self.patience == inf:
            return "Early stopping disabled, model performance may degrade over time"
        return f"Early stopping enabled, patience is set to {self.patience} epochs"
