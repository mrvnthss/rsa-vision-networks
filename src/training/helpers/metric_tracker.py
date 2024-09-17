"""A class to track performance metrics during training."""


import logging
from typing import Dict, List, Literal

import torch
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric


class MetricTracker:
    """A tracker to track performance metrics during training.

    The MetricTracker class is intended to track performance metrics
    during training at two different levels: over the entire duration of
    an epoch and for segments within an epoch.  This allows to log
    average results to TensorBoard for each segment within an epoch, as
    well as to report the total results at the end of an epoch.

    Note:
        This implementation assumes that all metrics in the
        ``prediction_metrics`` MetricCollection being passed to
        initialize an instance of this class accept two tensors
        "predictions" and "targets" as inputs, and return a scalar float
        tensor (so that the ``item`` method can be called on the tensor
        to obtain a scalar value).  E.g., a metric that can be tracked,
        is ``BinaryAccuracy`` with the ``multidim_average`` parameter
        set to "global".  In contrast, a metric that cannot be tracked
        is ``BinaryConfusionMatrix``, as it returns a 2x2 matrix.  When
        the ``compute_prediction_metrics`` method of this class is
        called for the first time, it automatically checks the output of
        all prediction metrics to ensure that they return a scalar float
        tensor.  Any metric that fails to do so is no longer tracked,
        and a warning is raised to inform the user about this.

    Attributes:
        device: The device to perform computations on.
        mean_metrics_partial: Same metrics as ``mean_metrics_total``,
          but reset at specified intervals within an epoch.
        mean_metrics_total: The mean metrics computed over the entire
          duration of an epoch.
        logger: The logger instance to record logs.
        prediction_metrics_partial: Same metrics as
          ``prediction_metrics_total``, but reset at specified intervals
          within an epoch.
        prediction_metrics_total: The prediction metrics computed over
          the entire duration of an epoch.

    Methods:
        compute_mean_metrics(level): Compute the mean metrics given
          their current state.
        compute_prediction_metrics(level): Compute the prediction
          metrics given their current state.
        report_status(): Report the metrics that are tracked during
          training.
        reset(partial, total): Reset metric states to their default
          values.
        update(mean_values, predictions, targets): Update the state
          variables of all metrics.
    """

    def __init__(
            self,
            mean_metrics: List[str],
            prediction_metrics: MetricCollection,
            device: torch.device
    ) -> None:
        """Initialize the MetricTracker instance.

        Args:
            mean_metrics: A list of names of metrics whose mean value is
              to be tracked during training.  These names are used to
              update the respective MeanMetric instances via the
              ``update`` method of this class.  See also the
              ``mean_values`` parameter of the ``update`` method.
            prediction_metrics: Metrics to track during training that
              are computed from the model predictions and target values.
              Each metric must accept two tensors "predictions" and
              "targets" as inputs, and return a scalar float tensor.
            device: The device to perform computations on.  This must be
              the same device that the inputs to the metrics are stored
              on.
        """

        self.mean_metrics_partial = {}
        self.mean_metrics_total = {}
        for name in mean_metrics:
            self.mean_metrics_partial[name] = MeanMetric().to(device)
            self.mean_metrics_total[name] = MeanMetric().to(device)

        prediction_metrics.reset()
        self.prediction_metrics_partial = prediction_metrics.to(device)
        self.prediction_metrics_total = prediction_metrics.clone().to(device)

        self.device = device

        self._prediction_metrics_checked = False

        self.logger = logging.getLogger(__name__)

    def compute_mean_metrics(
            self,
            level: Literal["partial", "total"]
    ) -> Dict[str, float]:
        """Compute the mean metrics given their current state.

        Args:
            level: The level at which to compute the mean metrics.  This
              can be either "partial" or "total".

        Returns:
            The computed mean metrics.

        Raises:
            ValueError: If ``level`` is neither "partial" nor "total".
        """

        if level == "partial":
            metric_values = {
                name: metric.compute().item() for name, metric in self.mean_metrics_partial.items()
            }
        elif level == "total":
            metric_values = {
                name: metric.compute().item() for name, metric in self.mean_metrics_total.items()
            }
        else:
            raise ValueError(
                f"'level' should be either 'partial' or 'total', but got {level}."
            )

        return metric_values

    def compute_prediction_metrics(
            self,
            level: Literal["partial", "total"]
    ) -> Dict[str, float]:
        """Compute the prediction metrics given their current state.

        Args:
            level: The level at which to compute the prediction metrics.
              This can be either "partial" or "total".

        Returns:
            The computed prediction metrics.

        Raises:
            ValueError: If ``level`` is neither "partial" nor "total".
        """

        def is_scalar(tensor: torch.Tensor) -> bool:
            """Check if a tensor is a scalar tensor."""

            return tensor.squeeze().dim() == 0

        # Compute prediction metrics
        if level == "partial":
            metric_values = self.prediction_metrics_partial.compute()
        elif level == "total":
            metric_values = self.prediction_metrics_total.compute()
        else:
            raise ValueError(
                f"'level' should be either 'partial' or 'total', but got {level}."
            )

        # Convert tensors to scalars
        if not self._prediction_metrics_checked:
            metric_values = {
                k: v.item() if is_scalar(v) else None for k, v in metric_values.items()
            }

            # Remove metrics that do not return a scalar value, if any
            invalid_metrics = [k for k, v in metric_values.items() if v is None]
            if len(invalid_metrics) > 0:
                self.logger.warning(
                    "The following metrics do not return a scalar value: %s. These metrics will "
                    "no longer be tracked.",
                    ", ".join(invalid_metrics)
                )

                # Extract valid metrics
                valid_partial_metrics = {
                    k: self.prediction_metrics_partial[k] for k in self.prediction_metrics_partial
                    if k not in invalid_metrics
                }
                valid_total_metrics = {
                    k: self.prediction_metrics_total[k] for k in self.prediction_metrics_total
                    if k not in invalid_metrics
                }

                # Replace MetricCollections accordingly
                self.prediction_metrics_partial = MetricCollection(
                    valid_partial_metrics
                ).to(self.device)
                self.prediction_metrics_total = MetricCollection(
                    valid_total_metrics
                ).to(self.device)

                # Remove None values from ``metric_values`` dictionary
                metric_values = {
                    k: v for k, v in metric_values.items() if v is not None
                }

            # Update flag to perform this check only once
            self._prediction_metrics_checked = True
        else:
            metric_values = {k: v.item() for k, v in metric_values.items()}

        return metric_values

    def report_status(self) -> None:
        """Report the metrics that are tracked during training."""

        all_metrics = [*self.mean_metrics_total.keys(), *self.prediction_metrics_total.keys()]

        if len(all_metrics) == 0:
            self.logger.info(
                "No metrics are being tracked during training."
            )
        else:
            self.logger.info(
                "Metrics being tracked during training are: %s.",
                ", ".join(all_metrics)
            )

    def reset(
            self,
            partial: bool = False,
            total: bool = False
    ) -> None:
        """Reset metric states to their default values.

        Args:
            partial: A flag to indicate whether to reset the partial
              metrics.
            total: A flag to indicate whether to reset the total
              metrics.
        """

        if partial:
            for metric in self.mean_metrics_partial.values():
                metric.reset()
            self.prediction_metrics_partial.reset()
        if total:
            for metric in self.mean_metrics_total.values():
                metric.reset()
            self.prediction_metrics_total.reset()

    def update(
            self,
            mean_values: Dict[str, float],
            predictions: torch.Tensor,
            targets: torch.Tensor,
    ) -> None:
        """Update the state variables of all metrics.

        Note:
            This method should be called after each mini-batch.  The
            values in the ``mean_values`` dictionary are weighted by the
            number of samples in the mini-batch.

        Args:
            mean_values: A dictionary containing the names of the mean
              metrics being tracked and the values to update them with.
            predictions: The model predictions.
            targets: The target values.
        """

        num_samples = len(targets)

        for name, value in mean_values.items():
            self.mean_metrics_partial[name].update(value=value, weight=num_samples)
            self.mean_metrics_total[name].update(value=value, weight=num_samples)

        self.prediction_metrics_partial.update(predictions, targets)
        self.prediction_metrics_total.update(predictions, targets)
