"""A base class to be subclassed by all trainers in this project."""


import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from src.base_classes.base_sampler import BaseSampler
from src.training.helpers.checkpoint_manager import CheckpointManager
from src.training.helpers.experiment_tracker import ExperimentTracker
from src.training.helpers.metric_tracker import MetricTracker
from src.training.helpers.performance_tracker import PerformanceTracker


class BaseTrainer(ABC):
    """A base trainer pooling common training functionality.

    Attributes:
        checkpoint_manager: The CheckpointManager instance responsible
          for saving and loading model checkpoints.
        device: The device to train on.
        epoch_idx: The current epoch index, starting from 1.
        experiment_tracker: The ExperimentTracker instance to log
          results to TensorBoard.
        logger: The logger instance to record logs.
        lr_scheduler: The scheduler used to adjust the learning rate
          during training.
        model: The model to be trained.
        num_epochs: The total number of epochs to train for.
        optimizer: The optimizer used during training.
        performance_tracker: The PerformanceTracker instance to monitor
          model performance and handle early stopping.
        preparation_time: A timestamp indicating the end of preparing a
          mini-batch.
        processing_time: A timestamp indicating the end of processing a
          mini-batch.
        start_time: A timestamp indicating the start of processing a
          mini-batch.
        train_loader: The dataloader providing training samples.
        val_loader: The dataloader providing validation samples.

    Methods:
        eval_compute_efficiency(): Evaluate the compute efficiency for
          an individual mini-batch.
        eval_epoch(): Evaluate the model on the validation set for a
          single epoch.
        get_global_step(is_training, batch_idx, batch_size): Get the
          global step (in # samples) in the training process.
        get_pbar(dataloader, mode): Wrap the provided dataloader with a
          progress bar.
        record_timestamp(stage): Record timestamp to monitor compute
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
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            device: torch.device,
            cfg: DictConfig,
            lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
            run_id: Optional[int] = None
    ) -> None:
        """Initialize the BaseTrainer instance.

        Note:
            The BaseTrainer instance passes the training configuration
            ``cfg`` to the CheckpointManager class during
            initialization.  Additionally, it makes direct use of the
            following entries of the configuration ``cfg``:
              * checkpoints.save_best_model
              * dataloader.batch_size
              * paths.tensorboard
              * performance.evaluate_on
              * performance.evaluation_metric
              * performance.higher_is_better
              * performance.keep_previous_best_score
              * performance.min_delta
              * performance.patience
              * tensorboard.updates_per_epoch
              * training.num_epochs
              * training.resume_from

        Args:
            model: The model to be trained.
            optimizer: The optimizer used during training.
            train_loader: The dataloader providing training samples.
            val_loader: The dataloader providing validation samples.
            device: The device to train on.
            cfg: The training configuration.
            lr_scheduler: The scheduler used to adjust the learning rate
              during training.
            run_id: Optional run ID to distinguish multiple runs using
              the same configuration.  Used to save checkpoints and
              event files in separate directories.
        """

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr_scheduler = lr_scheduler

        self.logger = logging.getLogger(__name__)

        self.num_epochs = cfg.training.num_epochs

        # CheckpointManager
        self.checkpoint_manager = CheckpointManager(
            cfg=cfg,
            run_id=run_id
        )
        self.checkpoint_manager.report_status()

        # PerformanceTracker
        if cfg.performance.evaluate_on not in ["train", "val"]:
            raise ValueError(
                "The dataset on which to evaluate the performance metric should be either 'train' "
                f"or 'val', but got {cfg.performance.evaluate_on}."
            )
        self.performance_metric = {
            "evaluation_metric": cfg.performance.evaluation_metric,
            "evaluate_on": cfg.performance.evaluate_on
        }
        self.performance_tracker = PerformanceTracker(
            higher_is_better=cfg.performance.higher_is_better,
            track_for_checkpointing=cfg.checkpoints.save_best_model,
            min_delta=cfg.performance.min_delta,
            patience=cfg.performance.patience
        )
        self.performance_tracker.report_status()

        # ExperimentTracker
        log_dir = cfg.paths.tensorboard
        if run_id is not None:
            log_dir = str(Path(log_dir) / f"run{run_id}")
        self.experiment_tracker = ExperimentTracker(
            log_dir=log_dir,
            updates_per_epoch=cfg.tensorboard.updates_per_epoch,
            batch_size=cfg.dataloader.batch_size,
            num_train_samples=self._get_num_samples("train"),
            num_val_samples=self._get_num_samples("val")
        )
        self.experiment_tracker.report_status()

        # Resume training from checkpoint
        if cfg.training.resume_from is not None:
            self.epoch_idx = self.checkpoint_manager.resume_training(
                resume_from=cfg.training.resume_from,
                device=self.device,
                model=self.model,
                optimizer=self.optimizer,
                performance_tracker=self.performance_tracker,
                keep_previous_best_score=cfg.performance.keep_previous_best_score,
                lr_scheduler=self.lr_scheduler
            )
            if self.epoch_idx > self.num_epochs:
                raise ValueError(
                    "The total number of epochs to train for should be larger than the number of "
                    "epochs the model has already been trained for!"
                )
        else:
            self.epoch_idx = 1

        # Update training sampler for deterministic shuffling
        self._update_training_sampler()

        # Initialize timestamps to track compute efficiency
        self.start_time = 0.
        self.preparation_time = 0.
        self.processing_time = 0.

    def eval_compute_efficiency(self) -> float:
        """Evaluate the compute efficiency for an individual mini-batch.

        Returns:
            The compute efficiency for an individual mini-batch.
        """

        total_time = self.processing_time - self.start_time
        processing_time = self.processing_time - self.preparation_time
        return processing_time / total_time

    def eval_epoch(self) -> Dict[str, float]:
        """Evaluate the model on the validation set for a single epoch.

        Returns:
            A dictionary containing the average loss and additional
            metrics computed during evaluation.
        """

        return self._run_epoch(is_training=False)

    def get_global_step(
            self,
            is_training: bool,
            batch_idx: int,
            batch_size: int
    ) -> int:
        """Get the global step (in # samples) in the training process.

        Args:
            is_training: Whether the model is being trained or
              evaluated.
            batch_idx: The index of the current mini-batch.
            batch_size: The size of the current mini-batch.  This may be
              smaller than the batch size specified in the dataloader if
              the number of samples is not divisible by the batch size.

        Returns:
            The global step (in number of samples processed) in the
            training process.
        """

        dataloader = self.train_loader if is_training else self.val_loader
        total_samples = len(dataloader.sampler) if hasattr(dataloader, "sampler") \
            else len(dataloader.dataset)

        step = ((self.epoch_idx - 1) * total_samples  # Completed epochs
                + batch_idx * dataloader.batch_size   # Full mini-batches
                + batch_size)                         # Current (partial) mini-batch

        return step

    def get_pbar(
            self,
            dataloader: torch.utils.data.DataLoader,
            mode: Literal["train", "val"] = "train"
    ) -> tqdm:
        """Wrap the provided dataloader with a progress bar.

        Args:
            dataloader: The dataloader to wrap with a progress bar.
            mode: The mode of the model to be displayed in the progress
              bar description, either "train" or "val".

        Returns:
            The provided dataloader wrapped with a progress bar.

        Raises:
            ValueError: If ``mode`` is neither "train" nor "val".
        """

        if mode not in ["train", "val"]:
            raise ValueError(f"'mode' should be either 'train' or 'val', but got {mode}.")

        # Construct description for progress bar
        num_digits = len(str(self.num_epochs))
        desc = (
            f"Epoch [{self.epoch_idx:0{num_digits}d}/{self.num_epochs}]    "
            f"{mode.capitalize()}"
        )

        # Wrap dataloader with progress bar
        pbar = tqdm(
            dataloader,
            desc=desc,
            leave=False,
            unit="batch"
        )
        return pbar

    def record_timestamp(
            self,
            stage: Literal["start", "preparation", "processing"]
    ) -> None:
        """Record timestamp to monitor compute efficiency.

        Args:
            stage: The stage of processing a mini-batch.  This should be
              either "start", "preparation", or "processing".

        Raises:
            ValueError: If ``stage`` is not one of "start",
              "preparation", or "processing".
        """

        timestamp = time.time()

        match stage:
            case "start":
                self.start_time = timestamp
            case "preparation":
                self.preparation_time = timestamp
            case "processing":
                self.processing_time = timestamp
            case _:
                raise ValueError(
                    "'stage' should be either 'start', 'preparation', or 'processing', "
                    f"but got {stage}."
                )

    def train(self) -> None:
        """Train the model for multiple epochs."""

        self.logger.info("Starting training loop ...")
        while self.epoch_idx <= self.num_epochs:
            # Train and validate the model, and update learning rate
            training_results = self.train_epoch()
            validation_results = self.eval_epoch()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Update PerformanceTracker
            if self.performance_tracker.is_tracking:
                results = training_results if self.performance_metric["evaluate_on"] == "train" \
                    else validation_results

                if self.performance_metric["evaluation_metric"] not in results:
                    raise ValueError(
                        f"Metric '{self.performance_metric['evaluation_metric']}' not found in "
                        f"training results. Please ensure that the metric is included in the "
                        f"dictionary returned by the 'train_epoch' and 'eval_epoch' methods."
                    )

                self.performance_tracker.update(
                    latest_score=results[self.performance_metric["evaluation_metric"]],
                    epoch_idx=self.epoch_idx
                )

            # Log results
            self._log_results(
                training_results=training_results,
                validation_results=validation_results
            )

            # Check for early stopping
            if self.performance_tracker.track_for_early_stopping:
                if self.performance_tracker.is_patience_exceeded():
                    self.experiment_tracker.close()
                    self.logger.info(
                        "No new best performance over the last %d consecutive epochs, stopping "
                        "training now.",
                        self.performance_tracker.patience
                    )
                    self._report_results()
                    return

            # Save checkpoint of best performing model, if applicable
            if self.performance_tracker.track_for_checkpointing:
                if self.performance_tracker.latest_is_best:
                    self.checkpoint_manager.save_checkpoint(
                        epoch_idx=self.epoch_idx,
                        model_state_dict=self.model.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        lr_scheduler_state_dict=self._get_lr_scheduler_state_dict(),
                        best_score=self.performance_tracker.best_score,
                        best_epoch_idx=self.performance_tracker.best_epoch_idx,
                        is_regular_save=False
                    )

            # Save regular checkpoint, if applicable
            if self.epoch_idx % self.checkpoint_manager.save_frequency == 0:
                self.checkpoint_manager.save_checkpoint(
                    epoch_idx=self.epoch_idx,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    lr_scheduler_state_dict=self._get_lr_scheduler_state_dict(),
                    best_score=self.performance_tracker.best_score,
                    best_epoch_idx=self.performance_tracker.best_epoch_idx,
                    is_regular_save=True
                )

            self.epoch_idx += 1
            self._update_training_sampler()

        # End of training
        self.experiment_tracker.close()
        self.logger.info("Training completed successfully.")
        self._report_results()

    def train_epoch(self) -> Dict[str, float]:
        """Train the model for a single epoch.

        Returns:
            A dictionary containing the average loss and additional
            metrics computed during training.
        """

        return self._run_epoch(is_training=True)

    def update_pbar_and_log_metrics(
            self,
            metric_tracker: MetricTracker,
            pbar: tqdm,
            batch_idx: int,
            mode: Literal["train", "val"],
            batch_size: int
    ) -> None:
        """Update progress bar and log metrics to TensorBoard.

        Args:
            metric_tracker: The MetricTracker instance to track
              performance metrics during training.
            pbar: The progress bar to update.
            batch_idx: The index of the current mini-batch.
            mode: Whether the model is being trained ("train") or
              evaluated ("val").
            batch_size: The size of the current mini-batch.
        """

        # Update progress bar
        mean_metrics_partial = metric_tracker.compute_mean_metrics("partial")
        prediction_metrics_partial = metric_tracker.compute_prediction_metrics(
            "partial"
        )
        all_metrics = {
            **mean_metrics_partial,
            **prediction_metrics_partial,
            "ComputeEfficiency": self.eval_compute_efficiency()
        }
        pbar.set_postfix(all_metrics)

        # Log to TensorBoard
        if self.experiment_tracker.is_tracking:
            # NOTE: The ``log_indices`` of the ExperimentTracker instance start from 1.
            if batch_idx + 1 in self.experiment_tracker.log_indices[mode]:
                all_metrics.pop("ComputeEfficiency")
                is_training = mode == "train"
                self.experiment_tracker.log_scalars(
                    scalars=all_metrics,
                    step=self.get_global_step(
                        is_training=is_training,
                        batch_idx=batch_idx,
                        batch_size=batch_size
                    ),
                    mode=mode
                )

                # Reset metrics for next set of mini-batches
                metric_tracker.reset(partial=True)

    @abstractmethod
    def _run_epoch(
            self,
            is_training: bool
    ) -> Dict[str, float]:
        """Run a single epoch of training or validation.

        Note:
            This method modifies the model in place when ``is_training``
            is True.

        Args:
            is_training: Whether to train or evaluate the model.

        Returns:
            A dictionary containing the average loss and additional
            metrics computed over the epoch.
        """

    def _get_lr_scheduler_state_dict(self) -> Optional[Dict[str, Any]]:
        """Retrieve the learning rate scheduler's state dict.

        Returns:
            The learning rate scheduler's state dict (if available).
        """

        if self.lr_scheduler is not None:
            return self.lr_scheduler.state_dict()
        return None

    def _get_num_samples(
            self,
            mode: Literal["train", "val"]
    ) -> int:
        """Count the number of total samples provided by a dataloader.

        Args:
            mode: Which dataloader to evaluate, either "train" or "val".

        Returns:
            The number of samples iterated over by the dataloader.

        Raises:
            ValueError: If ``mode`` is not one of "train" or "val".
        """

        if mode not in ["train", "val"]:
            raise ValueError(f"'mode' should be either 'train' or 'val', but got {mode}.")

        dataloader = self.train_loader if mode == "train" else self.val_loader
        if hasattr(dataloader, "sampler"):
            return len(dataloader.sampler)
        return len(dataloader.dataset)

    def _log_results(
            self,
            training_results: Dict[str, float],
            validation_results: Dict[str, float]
    ) -> None:
        """Log the results of training or validation.

        Args:
            training_results: The training results of the current epoch.
            validation_results: The validation results of the current
              epoch.
        """

        training_results_formatted = [
            f"{metric}: {value:.3f}" for metric, value in training_results.items()
        ]
        validation_results_formatted = [
            f"{metric}: {value:.3f}" for metric, value in validation_results.items()
        ]
        self.logger.info(
            "EPOCH [%0*d/%d]    TRAIN: %s    VAL: %s",
            len(str(self.num_epochs)),
            self.epoch_idx,
            self.num_epochs,
            "  ".join(training_results_formatted),
            "  ".join(validation_results_formatted)
        )

    def _report_results(self) -> None:
        """Report and log the final results of training."""

        dataset_str = "training" if self.performance_metric["evaluate_on"] == "train" \
            else "validation"
        self.logger.info(
            "Best performing model achieved a score of %.3f (%s) on the %s set after %d epochs of "
            "training.",
            self.performance_tracker.best_score,
            self.performance_metric["evaluation_metric"],
            dataset_str,
            self.performance_tracker.best_epoch_idx
        )

    def _update_training_sampler(self) -> None:
        """Update the sampler's epoch for deterministic shuffling."""

        if isinstance(self.train_loader.sampler, BaseSampler):
            self.train_loader.sampler.set_epoch_idx(self.epoch_idx)
