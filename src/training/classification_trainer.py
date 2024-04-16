"""This module provides the ClassificationTrainer class."""


import time

from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .train_utils import compute_log_indices


class ClassificationTrainer:
    """A class for training a classification model in PyTorch.

    Parameters:
        model: The model to be trained.
        train_loader: The dataloader providing training samples.
        val_loader: The dataloader providing validation samples
        loss_fn: The loss function used for training.
        optimizer: The optimizer used for training.
        device: The device to train on.
        cfg: The training configuration.

    (Additional) Attributes:
        writer: The TensorBoard writer.
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
        self.cfg = cfg
        self.device = device
        self.writer = SummaryWriter(cfg.logging.tb_dir)

    def train(self) -> None:
        """Train the model for a specified number of epochs.

        This method handles the entire training process.  It visualizes
        the model architecture and sample images in TensorBoard, trains
        and validates the model for a number of epochs, and updates the
        epoch index after each epoch. It also sets the epoch of the
        training set sampler for deterministic shuffling, and it closes
        the TensorBoard writer after training has finished.

        Note:
            This method modifies the model in place.
        """
        # Visualize model architecture in TensorBoard
        inputs, _ = next(iter(self.train_loader))
        self.writer.add_graph(self.model, inputs.to(self.device))

        # Visualize sample images of first batch in TensorBoard
        self.writer.add_images("sample_train_images", inputs, 0)

        for _ in range(self.cfg.training.num_epochs):
            # Train and validate the model for one epoch
            self._train_one_epoch()
            self._validate()

            # Increase epoch index and update train_sampler for deterministic shuffling
            self.cfg.logging.epoch_index += 1
            self.train_loader.sampler.set_epoch(self.cfg.logging.epoch_index)

        # Close TensorBoard writer
        self.writer.close()

    def _train_one_epoch(self) -> None:
        """Train the model for one epoch."""
        self._run_epoch(self.train_loader, self.optimizer)

    def _validate(self) -> None:
        """Validate the model."""
        self._run_epoch(self.val_loader)

    def _run_epoch(
            self,
            dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer = None
    ) -> None:
        """Run one epoch of training or validation.

        This method handles one epoch of training or validation,
        depending on whether an optimizer is provided.  It computes the
        loss and accuracy for each batch, updates the model parameters
        if in training mode, and logs the running totals for loss and
        accuracy to TensorBoard.

        Args:
            dataloader: The dataloader providing training or validation
              samples.
            optimizer: The optimizer used for training.  If None, the
              method validates the model.

        Note:
            This method modifies the model in place if an optimizer is
              provided.
    """
        # Running totals to report progress to TensorBoard
        running_samples = 0
        running_loss = 0.
        running_correct = 0

        # Set training/evaluation mode
        is_training = optimizer is not None
        self.model.train(is_training)

        # Determine batch indices at which to log to TensorBoard
        log_indices = compute_log_indices(dataloader, self.cfg.logging.intra_epoch_updates)

        # Set tags for TensorBoard logging
        tag_loss = f"{'train' if is_training else 'val'}/loss"
        tag_acc = f"{'train' if is_training else 'val'}/acc"

        # Prepare progress bar
        desc = (f"Epoch [{self.cfg.logging.epoch_index + 1}/{self.cfg.training.num_epochs}]    "
                f"{'Train' if is_training else 'Val'}")
        pbar = tqdm(dataloader, desc=desc, leave=False, unit="batch")

        # Initialize timer
        start_time = time.time()

        # Disable gradients during evaluation
        with (torch.set_grad_enabled(is_training)):
            for batch_index, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Keep track of the number of samples
                samples = len(targets)
                running_samples += samples

                # Determine preparation time
                prep_time = time.time() - start_time

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                # Backward pass and optimization
                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Determine compute efficiency
                process_time = time.time() - start_time - prep_time
                compute_efficiency = process_time / (prep_time + process_time) * 100  # in pct

                # Accumulate loss
                running_loss += loss.item() * samples

                # Compute accuracy
                _, predictions = torch.max(outputs, 1)
                correct = (predictions == targets).sum().item()
                running_correct += correct

                # Update progress bar
                avg_batch_loss = running_loss / running_samples
                avg_batch_acc = (running_correct / running_samples) * 100  # in pct
                pbar.set_postfix(
                    loss=avg_batch_loss,
                    accuracy=avg_batch_acc,
                    compute_efficiency=compute_efficiency
                )

                # Log batch loss and accuracy
                if batch_index in log_indices:
                    # Log to TensorBoard
                    global_step = self.cfg.logging.epoch_index * len(dataloader) + batch_index + 1
                    self.writer.add_scalar(tag_loss, avg_batch_loss, global_step)
                    self.writer.add_scalar(tag_acc, avg_batch_acc, global_step)

                    # Reset running totals
                    running_samples = 0
                    running_loss = 0.
                    running_correct = 0

                # Reset timer
                start_time = time.time()

        # Close progress bar
        pbar.close()

        # Flush writer after epoch for live updates
        self.writer.flush()
