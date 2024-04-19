"""A class to train a model for image classification in PyTorch."""


from omegaconf import DictConfig
import torch
from torch import nn

from src.utils import BalancedSampler, TrainingManager


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
        train_manager: An TrainingManager instance to perform auxiliary
          tasks during training and validation.
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

        self.train_manager = TrainingManager(
            self.model,
            self.train_loader,
            self.val_loader,
            self.device,
            self.cfg
        )

        self._update_train_sampler()

    def train(self) -> None:
        # Visualize model architecture in TensorBoard
        inputs, _ = next(iter(self.train_loader))
        self.train_manager.visualize_model(inputs.to(self.device))

        # Training loop
        for _ in range(self.cfg.training.num_epochs):
            self._train_one_epoch()
            self._validate()
            self._increment_epoch()
            self._update_train_sampler()

        # Close TensorBoard writer once training is finished
        self.train_manager.close_writer()

    def _train_one_epoch(self) -> None:
        self.train_manager.prepare_run("train")
        self._run_epoch()

    def _validate(self) -> None:
        self.train_manager.prepare_run("validate")
        self._run_epoch()

    def _run_epoch(self) -> None:
        """Run a single epoch of training or validation.

        Whether the model is in training or validation mode is
        determined by the ``self.train_manager.is_training`` attribute.
        Auxiliary tasks (i.e., switching between training and validation
        modes, updating the progress bar, logging metrics to
        TensorBoard, and computing the compute efficiency) are handled
        by the TrainingManager instance ``self.train_manager``.

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

    def _increment_epoch(self) -> None:
        self.train_manager.increment_epoch()

    def _update_train_sampler(self) -> None:
        """Update the sampler's epoch for deterministic shuffling."""
        if isinstance(self.train_loader.sampler, BalancedSampler):
            self.train_loader.sampler.set_epoch(self.train_manager.epoch)
