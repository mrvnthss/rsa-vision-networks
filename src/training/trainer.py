import time
from typing import List

from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
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
        # Visualize model architecture in TensorBoard
        inputs, _ = next(iter(self.train_loader))
        self.writer.add_graph(self.model, inputs.to(self.device))

        # Visualize sample images of first batch in TensorBoard
        self.writer.add_images("sample_train_images", inputs, 0)

        # Iteratively train and validate the model
        for _ in range(self.cfg.training.num_epochs):
            self._train_one_epoch()
            self._validate()
            self.cfg.logging.epoch_index += 1
            self.train_loader.sampler.set_epoch(self.cfg.logging.epoch_index)

        # Close TensorBoard writer
        self.writer.close()

    def _train_one_epoch(self) -> None:
        self._run_epoch(self.train_loader, self.optimizer)

    def _validate(self) -> None:
        self._run_epoch(self.val_loader)

    def _run_epoch(
            self,
            dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer = None
    ) -> None:
        # Running totals to report progress to TensorBoard
        running_samples = 0
        running_loss = 0.
        running_correct = 0

        # Set training/evaluation mode
        is_training = optimizer is not None
        self.model.train(is_training)

        # Determine batch indices at which to log to TensorBoard
        log_indices = self._compute_log_indices(dataloader)

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

    def _compute_log_indices(self, dataloader: torch.utils.data.DataLoader) -> List[int]:
        total_samples = len(dataloader.dataset)
        sample_intervals = torch.linspace(
            0, total_samples, self.cfg.logging.intra_epoch_updates + 1
        )
        log_indices = (
            torch.ceil(sample_intervals / self.cfg.dataloader.batch_size) - 1
        ).int().tolist()[1:]

        return log_indices
