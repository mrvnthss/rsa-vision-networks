"""A class to train a network for image classification in PyTorch."""


from typing import Dict, Literal, Optional

import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MetricCollection
from typing_extensions import override

from src.base_classes.base_trainer import BaseTrainer
from src.training.helpers.metric_tracker import MetricTracker


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
        lr_scheduler: The scheduler used to adjust the learning rate
          during training.
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
        update_pbar_and_log_metrics(metric_tracker, pbar, ...): Update
          progress bar and log metrics to TensorBoard.
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            prediction_metrics: MetricCollection,
            device: torch.device,
            cfg: DictConfig,
            lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
            run_id: Optional[int] = None
    ) -> None:
        """Initialize the ClassificationTrainer instance.

        Note:
            The ClassificationTrainer instance passes the training
            configuration ``cfg`` to the BaseTrainer class during
            initialization.

        Args:
            model: The model to be trained.
            optimizer: The optimizer used during training.
            criterion: The criterion used for optimization.
            train_loader: The dataloader providing training samples.
            val_loader: The dataloader providing validation samples.
            prediction_metrics: The metrics to track during training
              that are computed from the model predictions and target
              values.
            device: The device to train on.
            cfg: The training configuration.
            lr_scheduler: The scheduler used to adjust the learning rate
              during training.
            run_id: Optional run ID to distinguish multiple runs using
              the same configuration.  Used to save checkpoints and
              event files in separate directories.
        """

        # MetricTracker
        self.metric_tracker = MetricTracker(
            mean_metrics=["Loss"],
            prediction_metrics=prediction_metrics,
            device=device
        )
        self.metric_tracker.report_status()

        self.criterion = criterion

        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            cfg=cfg,
            lr_scheduler=lr_scheduler,
            run_id=run_id
        )

    @override
    def _run_epoch(
            self,
            is_training: bool
    ) -> Dict[str, float]:

        # Dataloader
        dataloader = self.train_loader if is_training else self.val_loader
        mode: Literal["train", "val"] = "train" if is_training else "val"
        wrapped_loader = self.get_pbar(dataloader, mode)

        # Reset MetricTracker
        self.metric_tracker.reset(partial=True, total=True)

        # Set model to appropriate mode
        self.model.train(is_training)

        # Loop over mini-batches
        with torch.set_grad_enabled(is_training):
            self.record_timestamp("start")
            for batch_idx, (inputs, targets) in enumerate(wrapped_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.record_timestamp("preparation")

                # Forward pass
                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)

                # Backward pass and optimization
                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.record_timestamp("processing")

                # Update MetricTracker
                self.metric_tracker.update(
                    mean_values={"Loss": loss.item()},
                    predictions=predictions,
                    targets=targets
                )

                # Update progress bar & log metrics to TensorBoard
                self.update_pbar_and_log_metrics(
                    metric_tracker=self.metric_tracker,
                    pbar=wrapped_loader,
                    batch_idx=batch_idx,
                    mode=mode,
                    batch_size=len(targets)
                )

                # Reset timer for next mini-batch
                self.record_timestamp("start")

        # Close progress bar and flush SummaryWriter
        wrapped_loader.close()
        self.experiment_tracker.flush()

        mean_metrics_total = self.metric_tracker.compute_mean_metrics("total")
        prediction_metrics_total = self.metric_tracker.compute_prediction_metrics("total")

        return {**mean_metrics_total, **prediction_metrics_total}
