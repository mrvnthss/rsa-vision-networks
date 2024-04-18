"""A class to train a model for image classification in PyTorch."""


from omegaconf import DictConfig
import torch
from torch import nn

from src.training.balanced_sampler import BalancedSampler
from src.utils import ExperimentTracker


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
        tracker: An ExperimentTracker instance to track loss and
          multiclass accuracy, and to reduce boilerplate code in the
          training loop.
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
        self.cfg = cfg

        self.tracker = ExperimentTracker(
            self.model, self.train_loader, self.val_loader, self.device, self.cfg
        )

        self._update_train_sampler()

    def train(self) -> None:
        # Visualize model architecture in TensorBoard
        inputs, _ = next(iter(self.train_loader))
        self.tracker.visualize_model(inputs.to(self.device))

        # Training loop
        for _ in range(self.cfg.training.num_epochs):
            self._train()
            self._validate()
            self._increment_epoch()
            self._update_train_sampler()

        # Close TensorBoard writer once training is finished
        self.tracker.close_writer()

    def _train(self) -> None:
        self.tracker.prepare_run("train")
        self._run_epoch()

    def _validate(self) -> None:
        self.tracker.prepare_run("validate")
        self._run_epoch()

    def _run_epoch(self) -> None:
        """Run a single epoch of training or validation.

        Whether the model is in training or validation mode is
        determined by the ``self.tracker.is_training`` attribute.
        Supporting tasks (i.e., tracking training metrics, updating
        the progress bar, and logging metrics to TensorBoard) are
        handled by the ExperimentTracker instance ``self.tracker``.

        Note:
            This method modifies the model in place when training.
        """
        pbar = self.tracker.get_pbar()

        # Loop over mini-batches
        with (torch.set_grad_enabled(self.tracker.is_training)):
            # Initial timestamp
            self.tracker.take_time("start")

            for features, targets in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)
                batch_size = len(targets)

                # Timestamp to compute preparation time
                self.tracker.take_time("prep")

                # Forward pass
                predictions = self.model(features)
                loss = self.loss_fn(predictions, targets)

                # Backward pass and optimization
                if self.tracker.is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Timestamp to compute processing time
                self.tracker.take_time("proc")

                # Update loss, multiclass accuracy, and progress bar
                self.tracker.update_loss(loss.item(), batch_size)
                self.tracker.update_mca(predictions, targets)
                self.tracker.update_pbar(pbar)

                # Add metrics to TensorBoard and increment batch index
                self.tracker.log_metrics()
                self.tracker.increment_batch()

                # Reset starting timestamp for next mini-batch
                self.tracker.take_time("start")

        # Close progress bar
        pbar.close()

        # Flush writer after epoch for live updates
        self.tracker.flush_writer()

    def _increment_epoch(self) -> None:
        self.tracker.increment_epoch()

    def _update_train_sampler(self) -> None:
        """Update the sampler's epoch for deterministic shuffling."""
        if isinstance(self.train_loader.sampler, BalancedSampler):
            self.train_loader.sampler.set_epoch(self.tracker.epoch)
