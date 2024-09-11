"""A class to track performance metrics during training."""


import logging
from typing import Dict, Literal

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
        This implementation assumes that all metrics in the ``metrics``
        MetricCollection being passed to initialize an instance of this
        class accept two tensors 'predictions' and 'targets' as inputs,
        and return a tensor consisting of a single element (so that the
        ``item`` method can be called on the tensor to return a scalar
        value).  A metric that can be tracked, for example, is
        ``BinaryAccuracy`` with the ``multidim_average`` parameter set
        to "global".  In contrast, a metric that cannot be tracked is
        ``BinaryConfusionMatrix``, as it returns a 2x2 matrix.  When the
        ``compute`` method of this class is called for the first time,
        it will check the output of the metrics to ensure that they
        return a scalar value.  If any metric does not return a scalar
        value, a warning will be logged, and that metric will no longer
        be tracked.

    Attributes:
        device: The device to perform metric computations on.
        logger: The logger instance to record logs.
        partial_loss: The average loss computed over the current segment
          (i.e., since metrics were last logged to TensorBoard).
        partial_metrics: The same metrics as ``total_metrics``, but
          reset at specified intervals within an epoch.
        total_loss: The average loss computed over the entire duration
          of an epoch.
        total_metrics: The metrics computed over the entire duration of
          an epoch.

    Methods:
        compute(level): Compute the metric values given the current
          state.
        report_status(): Report the metrics that are tracked during
          training.
        reset(partial, total): Reset metric states to their default
          values.
        update(predictions, targets, loss): Update the state variables
          of all metrics.
    """

    def __init__(
            self,
            metrics: MetricCollection,
            device: torch.device
    ) -> None:
        """Initialize the MetricTracker instance.

        Args:
            metrics: The metrics to track during training.
            device: The device to perform metric computations on.  This
              must be the same device that the inputs to the metrics are
              stored on.
        """

        metrics.reset()
        self.partial_metrics = metrics.to(device)
        self.total_metrics = metrics.clone().to(device)

        self.partial_loss = MeanMetric().to(device)
        self.total_loss = MeanMetric().to(device)

        self.device = device

        self._metric_outputs_checked = False

        self.logger = logging.getLogger(__name__)

    def compute(
            self,
            level: Literal["partial", "total"]
    ) -> Dict[str, float]:
        """Compute the metric values given the current state.

        Args:
            level: The level at which to compute the metric values,
              including the average loss.  This can be either "partial"
              or "total".

        Returns:
            The computed metric values including the average loss value.

        Raises:
            ValueError: If ``level`` is neither "partial" nor "total".
        """

        def is_scalar(tensor: torch.Tensor) -> bool:
            """Check if a tensor is a scalar."""

            return tensor.squeeze().dim() == 0

        # Compute metric values and average loss
        if level == "partial":
            metric_values = self.partial_metrics.compute()
            avg_loss = self.partial_loss.compute().item()
        elif level == "total":
            metric_values = self.total_metrics.compute()
            avg_loss = self.total_loss.compute().item()
        else:
            raise ValueError(
                f"'level' should be either 'partial' or 'total', but got {level}."
            )

        # Convert metric values from tensors to scalars
        if not self._metric_outputs_checked:
            metric_values = {
                k: v.item() if is_scalar(v) else None for k, v in metric_values.items()
            }

            # Check for metrics that do not return a scalar value
            invalid_metrics = [k for k, v in metric_values.items() if v is None]
            if len(invalid_metrics) > 0:
                self.logger.warning(
                    "The following metrics do not return a scalar value: %s. These metrics will "
                    "no longer be tracked.",
                    ", ".join(invalid_metrics)
                )

                # Extract valid metrics
                valid_partial_metrics = {
                    k: self.partial_metrics[k] for k in self.partial_metrics
                    if k not in invalid_metrics
                }
                valid_total_metrics = {
                    k: self.total_metrics[k] for k in self.total_metrics
                    if k not in invalid_metrics
                }

                # Replace MetricCollections accordingly
                self.partial_metrics = MetricCollection(valid_partial_metrics).to(self.device)
                self.total_metrics = MetricCollection(valid_total_metrics).to(self.device)

                # Remove None values from ``metric_values`` dictionary
                metric_values = {
                    k: v for k, v in metric_values.items() if v is not None
                }

            # Set flag to perform this check only once
            self._metric_outputs_checked = True
        else:
            metric_values = {k: v.item() for k, v in metric_values.items()}

        return {
            "Loss": avg_loss,
            **metric_values
        }

    def report_status(self) -> None:
        """Report the metrics that are tracked during training."""

        if len(self.total_metrics) == 0:
            self.logger.info(
                "No additional metrics are being tracked during training besides loss."
            )
        else:
            self.logger.info(
                "Additional metrics being tracked during training: %s.",
                ", ".join(self.total_metrics.keys())
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
            self.partial_metrics.reset()
            self.partial_loss.reset()
        if total:
            self.total_metrics.reset()
            self.total_loss.reset()

    def update(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            loss: torch.Tensor
    ) -> None:
        """Update the state variables of all metrics.

        Args:
            predictions: The model predictions.
            targets: The target values.
            loss: The loss value.
        """

        num_samples = len(targets)
        loss_value = loss.item()

        self.partial_metrics.update(predictions, targets)
        self.partial_loss.update(loss_value, num_samples)

        self.total_metrics.update(predictions, targets)
        self.total_loss.update(loss_value, num_samples)
