"""A class to track model performance in PyTorch."""


from typing import Dict

from numpy import inf


class PerformanceTracker:
    """A tracker to monitor model performance in PyTorch.

    Params:
        metrics: A dictionary of metrics to track.
        performance_metric: The metric to evaluate model performance.
        higher_is_better: A flag to indicate whether higher values
          of the ``performance_metric`` reflect better performance.
        patience: The number of consecutive epochs without an increase
          in performance to wait before stopping the training process.
          Set to ``inf`` to disable early stopping.

    (Additional) Attributes:
        best_score: The best score of the ``performance_metric``
          observed so far.
        latest_is_best: A flag to indicate whether the latest score
          of the ``performance_metric`` is the best observed so far.
        patience_counter: The number of consecutive epochs without an
          increase in performance.
    """

    def __init__(
            self,
            metrics: Dict[str, float],
            performance_metric: str,
            higher_is_better: bool,
            patience: int
    ) -> None:
        self.metrics = metrics
        self.performance_metric = performance_metric
        self.higher_is_better = higher_is_better
        self.patience = patience

        assert self.performance_metric in self.metrics

        self.best_score = -inf if self.higher_is_better else inf
        self.latest_is_best = False
        self.patience_counter = 0

    def update(
            self,
            metrics: Dict[str, float]
    ) -> None:
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
        return self.patience_counter >= self.patience

    def get_status(self) -> str:
        if self.patience == inf:
            return "Early stopping disabled, model performance may degrade over time"
        else:
            return f"Early stopping enabled, patience is set to {self.patience} epochs"
