"""A class to train a model for image classification in PyTorch."""


import logging
from typing import Tuple

from numpy import inf
from omegaconf import DictConfig
import torch
from torch import nn

from src.utils import BalancedSampler, CheckpointManager, PerformanceTracker, TrainingManager


class ClassificationTrainer:
    """A trainer to train a classification model in PyTorch.

    Params:
        model: The model to be trained.
        train_loader: The dataloader providing training samples.
        val_loader: The dataloader providing validation samples.
        loss_fn: The loss function used for training.
        optimizer: The optimizer used for training.
        device: The device to train on.
        cfg: The training configuration.

    (Additional) Attributes:
        num_epochs: The total number of epochs to train for.
        save_periodically: Whether to periodically save model
          checkpoints.
        save_frequency: The frequency at which to periodically save
          model checkpoints.
        delete_previous: Whether to delete previous checkpoints when
          saving new ones.  Only applies to periodically saved
          checkpoints.
        save_best: Whether to continuously save the best performing
          model.
        saving_enabled: Whether to save model checkpoints at all.
        early_stopping: Whether to perform early stopping.
        tracking_enabled: Whether to track model performance for saving
          purposes or early stopping.
        train_manager: A TrainingManager instance to perform auxiliary
          tasks during training and validation.
        chkpt_manager: A CheckpointManager instance to save model
          checkpoints.  Only initialized if saving is enabled or
          training is to be resumed from a checkpoint, else set to None.
        performance_tracker: A PerformanceTracker instance to monitor
          model performance and handle early stopping.  Only initialized
          if tracking is enabled, else set to None.
        logger: A logger instance to record logs.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            loss_fn: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            cfg: DictConfig
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        self.num_epochs = cfg.training.num_epochs

        self.save_regularly = cfg.checkpoints.save_frequency > 0
        self.save_frequency = cfg.checkpoints.save_frequency
        self.delete_previous = cfg.checkpoints.delete_previous
        self.save_best = cfg.checkpoints.save_best
        self.saving_enabled = self.save_regularly or self.save_best

        self.early_stopping = cfg.checkpoints.patience > 0
        self.tracking_enabled = self.save_best or self.early_stopping

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize training manager
        self.train_manager = TrainingManager(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            cfg=cfg
        )

        # Checkpoint manager
        if self.saving_enabled or cfg.training.resume_from:
            # Initialize manager
            self.chkpt_manager = CheckpointManager(
                checkpoint_dir=cfg.checkpoints.checkpoint_dir,
                cfg=cfg
            )

            # Report status
            msgs = self.chkpt_manager.get_status(self.save_regularly, self.save_best)
            for msg in msgs:
                self.logger.info(msg)
        else:
            self.chkpt_manager = None
            self.logger.info("No checkpoints will be saved during training")

        # Performance tracker
        if self.tracking_enabled:
            # Set up parameters
            metrics = {
                "val_loss": 0.,
                "val_mca": 0.
            }
            higher_is_better = cfg.training.performance_metric == "val_mca"
            patience = cfg.checkpoints.patience if self.early_stopping else inf

            # Initialize tracker
            self.performance_tracker = PerformanceTracker(
                metrics=metrics,
                performance_metric=cfg.training.performance_metric,
                higher_is_better=higher_is_better,
                patience=patience
            )

            # Report status
            msg = self.performance_tracker.get_status()
            self.logger.info(msg)
        else:
            self.performance_tracker = None
            self.logger.info("Early stopping disabled, model performance may degrade over time")

        # Resume training if a checkpoint is provided
        if cfg.training.resume_from:
            self.chkpt_manager.resume_training(
                resume_from=cfg.training.resume_from,
                model=self.model,
                optimizer=self.optimizer,
                device=self.device,
                train_manager=self.train_manager,
                performance_tracker=self.performance_tracker
            )

        # Set the training sampler's epoch for deterministic shuffling
        self._update_train_sampler()

    def train(self) -> None:
        # Visualize model architecture in TensorBoard
        inputs, _ = next(iter(self.train_loader))
        self.train_manager.visualize_model(inputs.to(self.device))

        self.logger.info("Starting training loop")
        for _ in range(self.num_epochs):
            # Make sure that sampler of train_loader is in sync with train_manager
            assert (self.train_manager.epoch == self.train_loader.sampler.epoch)

            # Train and validate for one epoch
            train_loss, train_mca = self._train_one_epoch()
            val_loss, val_mca = self._validate()

            # Track performance metrics
            if self.tracking_enabled:
                metrics = {
                    "val_loss": val_loss,
                    "val_mca": val_mca
                }
                self.performance_tracker.update(metrics)

            # Report results
            self.logger.info(
                "Epoch [%0*d/%d]    Train  Loss: %.4f  MCA: %.2f    Val  Loss: %.4f  MCA: %.2f",
                len(str(self.train_manager.final_epoch)),
                self.train_manager.epoch,
                self.train_manager.final_epoch,
                train_loss,
                train_mca,
                val_loss,
                val_mca
            )

            # Check for early stopping
            if self.early_stopping:
                if self.performance_tracker.is_patience_exceeded():
                    # Close TensorBoard writer and stop training
                    self.train_manager.close_writer()
                    self.logger.info(
                        "Performance has not improved for %d consecutive epochs, "
                        "stopping training now",
                        self.performance_tracker.patience
                    )
                    return

            # Check for new best performance and possibly save checkpoint
            if self.save_best and self.performance_tracker.latest_is_best:
                self.chkpt_manager.save_checkpoint(
                    epoch=self.train_manager.epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    best_score=self.performance_tracker.best_score,
                    is_best=True
                )

            # Regular checkpoint save
            if self.save_regularly:
                if self.train_manager.epoch % self.save_frequency == 0:
                    best_score = (
                        self.performance_tracker.best_score if self.tracking_enabled else None
                    )
                    self.chkpt_manager.save_checkpoint(
                        epoch=self.train_manager.epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        best_score=best_score,
                        delete_previous=self.delete_previous
                    )

            # Increment epoch counter and update training sampler
            self._increment_epoch()
            self._update_train_sampler()

        # Close TensorBoard writer and log training completion
        self.train_manager.close_writer()
        self.logger.info("Training completed successfully")

    def _train_one_epoch(self) -> Tuple[float, float]:
        self.train_manager.prepare_run("train")
        train_loss, train_mca = self._run_epoch()
        return train_loss, train_mca

    def _validate(self) -> Tuple[float, float]:
        self.train_manager.prepare_run("validate")
        val_loss, val_mca = self._run_epoch()
        return val_loss, val_mca

    def _run_epoch(self) -> Tuple[float, float]:
        """Run a single epoch of training or validation.

        Whether the model is in training or validation mode is
        determined by the ``self.train_manager.is_training`` attribute.
        Auxiliary tasks (i.e., switching between training and validation
        modes, updating the progress bar, logging metrics to
        TensorBoard, and computing the compute efficiency) are handled
        by the TrainingManager instance ``self.train_manager``.

        Returns:
            The average loss and multiclass accuracy for the epoch.

        Note:
            This method modifies the model in place when training.
        """
        pbar = self.train_manager.get_pbar()

        # Loop over mini-batches
        with (torch.set_grad_enabled(self.train_manager.is_training)):
            # Initial timestamp
            self.train_manager.take_time("start")

            for features, targets in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)
                batch_size = len(targets)

                # Timestamp to compute preparation time
                self.train_manager.take_time("prep")

                # Forward pass
                predictions = self.model(features)
                loss = self.loss_fn(predictions, targets)

                # Backward pass and optimization
                if self.train_manager.is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Timestamp to compute processing time
                self.train_manager.take_time("proc")

                # Update loss, multiclass accuracy, and progress bar
                self.train_manager.update_loss(loss.item(), batch_size)
                self.train_manager.update_mca(predictions, targets)
                self.train_manager.update_pbar(pbar)

                # Add metrics to TensorBoard and increment batch index
                self.train_manager.log_metrics()
                self.train_manager.increment_batch()

                # Reset starting timestamp for next mini-batch
                self.train_manager.take_time("start")

        # Close progress bar
        pbar.close()

        # Flush writer after epoch for live updates
        self.train_manager.flush_writer()

        # Compute average loss and multiclass accuracy for the epoch
        loss = self.train_manager.compute_loss(total=True)
        mca = self.train_manager.compute_mca(total=True)

        return loss, mca

    def _increment_epoch(self) -> None:
        self.train_manager.increment_epoch()

    def _update_train_sampler(self) -> None:
        """Update the sampler's epoch for deterministic shuffling."""
        if isinstance(self.train_loader.sampler, BalancedSampler):
            self.train_loader.sampler.set_epoch(self.train_manager.epoch)
