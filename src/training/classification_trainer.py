"""A class to train a network for image classification in PyTorch."""


from collections import OrderedDict
from typing import Dict

import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MetricCollection
from typing_extensions import Literal

from src.base_classes import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    """A trainer to train a network for image classification in PyTorch.

    Attributes:
        checkpoint_manager: The CheckpointManager instance responsible
          for saving and loading model checkpoints.
        criterion: The criterion used for optimization.
        device: The device to train on.
        epoch_idx: The current epoch index, starting from 1.
        experiment_tracker: The ExperimentTracker instance to log
          results to TensorBoard.
        final_epoch_idx: The index of the final epoch.
        logger: The logger instance to record logs.
        metric_tracker: The MetricTracker instance to track performance
          metrics during training.
        model: The model to be trained.
        optimizer: The optimizer used during training.
        performance_tracker: The PerformanceTracker instance to monitor
          model performance and handle early stopping.
        preparation_time: A timestamp indicating the end of preparing a
          mini-batch (i.e., loading and moving to target device).
        processing_time: A timestamp indicating the end of processing a
          mini-batch.
        start_time: A timestamp indicating the start of processing a
          mini-batch.
        train_loader: The dataloader providing training samples.
        val_loader: The dataloader providing validation samples.

    Methods:
        eval_compute_efficiency(): Evaluate the compute efficiency for
          an individual batch.
        eval_epoch(): Evaluate the model on the validation set for a
          single epoch.
        get_pbar(dataloader, mode): Wrap the provided dataloader with a
          progress bar.
        record_timestamp(stage): Record timestamp to track compute
          efficiency.
        train(): Train the model for multiple epochs.
        train_epoch(): Train the model for a single epoch.
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            metrics: MetricCollection,
            device: torch.device,
            cfg: DictConfig
    ) -> None:
        """Initialize the ClassificationTrainer instance.

        Args:
            model: The model to be trained.
            optimizer: The optimizer used during training.
            criterion: The criterion used for optimization.
            train_loader: The dataloader providing training samples.
            val_loader: The dataloader providing validation samples.
            metrics: The additional metrics to track during training
              besides loss.
            device: The device to train on.
            cfg: The training configuration.
        """

        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            metrics=metrics,
            device=device,
            cfg=cfg
        )
        self.criterion = criterion

    def train_epoch(self) -> Dict[str, float]:
        """Train the model for a single epoch.

        Returns:
            A dictionary containing the average loss and additional
            metrics computed during training.
        """

        return self._run_epoch(is_training=True)

    def eval_epoch(self) -> Dict[str, float]:
        """Evaluate the model on the validation set for a single epoch.

        Returns:
            A dictionary containing the average loss and additional
            metrics computed during evaluation.
        """

        return self._run_epoch(is_training=False)

    def _run_epoch(
            self,
            is_training: bool
    ) -> Dict[str, float]:
        """Run a single epoch of training or validation.

        Args:
            is_training: Whether to train the model or evaluate it.

        Returns:
            A dictionary containing the average loss and additional
            metrics computed over the epoch.

        Note:
            This method modifies the model in place when training.
        """

        mode: Literal["Train", "Val"] = "Train" if is_training else "Val"

        # Dataloader
        dataloader = self.train_loader if is_training else self.val_loader
        wrapped_loader = self.get_pbar(dataloader, mode)

        # Reset MetricTracker
        self.metric_tracker.reset(partial=True, total=True)

        # Set model to appropriate mode
        self.model.train(is_training)

        # Loop over mini-batches
        with torch.set_grad_enabled(is_training):
            self.record_timestamp("start")
            for batch_idx, (features, targets) in enumerate(wrapped_loader):
                features = features.to(self.device)
                targets = targets.to(self.device)

                self.record_timestamp("preparation")

                # Forward pass
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)

                # Backward pass and optimization
                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.record_timestamp("processing")

                # Update MetricTracker
                self.metric_tracker.update(predictions, targets, loss)

                # Update progress bar
                ordered_metrics = OrderedDict()
                for metric, value in self.metric_tracker.compute("partial").items():
                    ordered_metrics[metric] = value
                ordered_metrics["ComputeEfficiency"] = self.eval_compute_efficiency()
                wrapped_loader.set_postfix(ordered_metrics)

                # Log to TensorBoard
                if self.experiment_tracker.is_tracking:
                    if batch_idx + 1 in self.experiment_tracker.log_indices[mode]:
                        # Global step (in number of batches, starting from 1)
                        step = (self.epoch_idx - 1) * len(dataloader) + (batch_idx + 1)

                        # Log metrics to TensorBoard
                        self.experiment_tracker.log_scalars(
                            scalars=self.metric_tracker.compute("partial"),
                            step=step,
                            mode=mode
                        )

                        # Reset metrics for next set of mini-batches
                        self.metric_tracker.reset(partial=True)

                # Reset timer for next mini-batch
                self.record_timestamp("start")

        # Close progress bar and flush SummaryWriter
        wrapped_loader.close()
        self.experiment_tracker.flush()

        return self.metric_tracker.compute("total")
