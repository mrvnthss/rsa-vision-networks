"""A base class to be subclassed by all trainers in this project."""


import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Literal, Optional

import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from src.base_classes.base_sampler import BaseSampler
from src.training.helpers.checkpoint_manager import CheckpointManager
from src.training.helpers.experiment_tracker import ExperimentTracker
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
        final_epoch_idx: The index of the final epoch.
        logger: The logger instance to record logs.
        model: The model to be trained.
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
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            device: torch.device,
            cfg: DictConfig,
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
              * performance.dataset
              * performance.higher_is_better
              * performance.keep_previous_best_score
              * performance.metric
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
            run_id: Optional run ID to distinguish multiple runs using
              the same configuration.  Used to save checkpoints and
              event files in separate directories.
        """

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.logger = logging.getLogger(__name__)

        # CheckpointManager
        self.checkpoint_manager = CheckpointManager(
            cfg=cfg,
            run_id=run_id
        )
        self.checkpoint_manager.report_status()

        # PerformanceTracker
        if cfg.performance.dataset not in ["Train", "Val"]:
            raise ValueError(
                "The dataset on which to evaluate the performance metric should be either 'Train' "
                f"or 'Val', but got {cfg.performance.dataset}."
            )
        self.performance_metric = {
            "metric": cfg.performance.metric,
            "dataset": cfg.performance.dataset
        }
        self.performance_tracker = PerformanceTracker(
            higher_is_better=cfg.performance.higher_is_better,
            track_for_checkpointing=cfg.checkpoints.save_best_model,
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
            num_train_samples=self._get_num_samples("Train"),
            num_val_samples=self._get_num_samples("Val")
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
                keep_previous_best_score=cfg.performance.keep_previous_best_score
            )
        else:
            self.epoch_idx = 1
        self.final_epoch_idx = self.epoch_idx + cfg.training.num_epochs - 1

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
            mode: Literal["Train", "Val"] = "Train"
    ) -> tqdm:
        """Wrap the provided dataloader with a progress bar.

        Args:
            dataloader: The dataloader to wrap with a progress bar.
            mode: The mode of the model to be displayed in the progress
              bar description, either "Train" or "Val".

        Returns:
            The provided dataloader wrapped with a progress bar.

        Raises:
            ValueError: If ``mode`` is not one of "Train" or "Val".
        """

        if mode not in ["Train", "Val"]:
            raise ValueError(f"'mode' should be either 'Train' or 'Val', but got {mode}.")

        # Construct description for progress bar
        num_digits = len(str(self.final_epoch_idx))
        desc = f"Epoch [{self.epoch_idx:0{num_digits}d}/{self.final_epoch_idx}]    {mode}"

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
        while self.epoch_idx <= self.final_epoch_idx:
            # Train and validate the model
            training_results = self.train_epoch()
            validation_results = self.eval_epoch()

            # Update PerformanceTracker
            if self.performance_tracker.is_tracking:
                results = training_results if self.performance_metric["dataset"] == "Train" \
                    else validation_results

                if self.performance_metric["metric"] not in results:
                    raise ValueError(
                        f"Performance metric '{self.performance_metric['metric']}' not found in "
                        "training results. Please ensure that the performance metric is included "
                        "in the dictionary returned by the 'train_epoch' and 'eval_epoch' methods."
                    )

                self.performance_tracker.update(
                    latest_score=results[self.performance_metric["metric"]],
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

    def _get_num_samples(
            self,
            mode: Literal["Train", "Val"]
    ) -> int:
        """Count the number of total samples provided by a dataloader.

        Args:
            mode: Which dataloader to evaluate, either "Train" or "Val".

        Returns:
            The number of samples iterated over by the dataloader.

        Raises:
            ValueError: If ``mode`` is not one of "Train" or "Val".
        """

        if mode not in ["Train", "Val"]:
            raise ValueError(f"'mode' should be either 'Train' or 'Val', but got {mode}.")

        dataloader = self.train_loader if mode == "Train" else self.val_loader
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
            len(str(self.final_epoch_idx)),
            self.epoch_idx,
            self.final_epoch_idx,
            "  ".join(training_results_formatted),
            "  ".join(validation_results_formatted)
        )

    def _report_results(self) -> None:
        """Report and log the final results of training."""

        dataset_str = "training" if self.performance_metric["dataset"] == "Train" \
            else "validation"
        self.logger.info(
            "Best performing model achieved a score of %.3f (%s) on the %s set after %d epochs of "
            "training.",
            self.performance_tracker.best_score,
            self.performance_metric["metric"],
            dataset_str,
            self.performance_tracker.best_epoch_idx
        )

    def _update_training_sampler(self) -> None:
        """Update the sampler's epoch for deterministic shuffling."""

        if isinstance(self.train_loader.sampler, BaseSampler):
            self.train_loader.sampler.set_epoch_idx(self.epoch_idx)
