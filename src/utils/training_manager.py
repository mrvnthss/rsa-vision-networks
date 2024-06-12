"""A class to handle auxiliary tasks during training and validation."""


import time

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm


class TrainingManager:
    """A manager to handle auxiliary tasks during training/validation.

    Handles auxiliary tasks during training and validation, such as
    switching between training and validation modes, updating the
    progress bar, logging data to TensorBoard, and computing the
    compute efficiency.

    Attributes:
        batch: The current batch number.
        device: The device to train on.
        epoch: The current epoch number.
        final_epoch: The final epoch number.
        is_training: A flag to indicate whether the model is training.
        model: The model to be trained.
        num_epochs: The total number of epochs to train for.
        prep_time: A timestamp indicating the end of preparing a
          mini-batch.
        proc_time: A timestamp indicating the end of processing a
          mini-batch.
        running_loss: The running loss during training/validation.
          Resets after each logging interval specified by
          ``tb_indices``.
        running_mca: A metric to track the multiclass accuracy during
          training/validation.  Resets after each logging interval
          specified by ``tb_indices``.
        running_samples: The running number of samples processed.
          Resets after each logging interval specified by
          ``tb_indices``.
        start_epoch: The starting epoch number.
        start_time: A timestamp indicating the start of processing a
          mini-batch.
        tb_indices: Batch indices at which to log data for consumption
          by TensorBoard.
        tb_tags: Tags for logging data to TensorBoard.
        tb_updates: The number of times per epoch to log data for
          consumption by TensorBoard.
        total_loss: The total loss during training/validation.  Resets
          after each epoch.
        total_mca: A metric to track the multiclass accuracy during
          training/validation.  Resets after each epoch.
        total_samples: The total number of samples processed.  Resets
          after each epoch.
        train_loader: The dataloader providing training samples.
        val_loader: The dataloader providing validation samples.
        writer: A SummaryWriter instance to log data for consumption and
          visualization by TensorBoard.

    Methods:
        close_writer(): Close the SummaryWriter instance.
        compute_loss(total): Compute the loss during
          training/validation.
        compute_mca(total): Compute the multiclass accuracy during
          training/validation.
        flush_writer(): Flush the SummaryWriter instance.
        get_pbar(): Get a progress bar for training or validation.
        increment_batch(): Increment the batch number.
        increment_epoch(): Increment the epoch number.
        log_scalars(): Log scalars to TensorBoard at specified batches.
        prepare_run(state): Perform setup before training/validation.
        take_time(stage): Record timestamp.
        update_loss(loss, batch_size): Update running and total loss.
        update_mca(preds, targets): Update running and total multiclass
          accuracy.
        update_pbar(pbar): Update the progress bar with the latest
          metrics.
        visualize_model(inputs): Visualize the model architecture in
          TensorBoard.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            device: torch.device,
            cfg: DictConfig
    ) -> None:
        """Initialize the TrainingManager instance.

        Args:
            model: The model to be trained.
            train_loader: The dataloader providing training samples.
            val_loader: The dataloader providing validation samples.
            device: The device to train on.
            cfg: The training configuration.
        """

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.writer = SummaryWriter(cfg.paths.logs)
        self.tb_updates = cfg.training.tb_updates
        self.tb_indices = []
        self.tb_tags = {}

        self.is_training = True

        self.num_epochs = cfg.training.num_epochs
        self.start_epoch = 1
        self.final_epoch = self.start_epoch + self.num_epochs - 1
        self.epoch = 1
        self.batch = 1

        self.running_samples = 0
        self.running_loss = 0.
        self.running_mca = MulticlassAccuracy(
            num_classes=cfg.dataset.num_classes,
            average="micro"
        ).to(self.device)

        self.total_samples = 0
        self.total_loss = 0.
        self.total_mca = MulticlassAccuracy(
            num_classes=cfg.dataset.num_classes,
            average="micro"
        ).to(self.device)

        self.start_time = 0.
        self.prep_time = 0.
        self.proc_time = 0.

    def prepare_run(
            self,
            state: str
    ) -> None:
        """Perform setup before training/validation.

        Set the model to training or evaluation mode, reset the batch
        index, compute indices at which to log metrics to TensorBoard,
        and set TensorBoard tags.

        Args:
            state: The desired state of the model, either "train" or
              "val".
        """

        self.is_training = state == "train"
        self.model.train(self.is_training)
        self.batch = 1
        self._reset_metrics(total=True)
        self._set_tb_indices()
        self.tb_tags = {
            "loss": f"loss/{'train' if self.is_training else 'val'}",
            "acc": f"acc/{'train' if self.is_training else 'val'}"
        }

    def visualize_model(
            self,
            inputs: torch.Tensor
    ) -> None:
        """Visualize the model architecture in TensorBoard.

        Args:
            inputs: A sample input tensor to pass through the model for
              visualization.
        """

        self.writer.add_graph(self.model, inputs)

    def get_pbar(self) -> tqdm:
        """Get a progress bar for training or validation.

        Returns:
            A progress bar for training or validation.
        """

        dataloader = self.train_loader if self.is_training else self.val_loader
        num_digits = len(str(self.final_epoch))
        mode = "Train" if self.is_training else "Val"
        desc = f"Epoch [{self.epoch:0{num_digits}d}/{self.final_epoch}]    {mode}"
        return tqdm(dataloader, desc=desc, leave=False, unit="batch")

    def update_pbar(
            self,
            pbar: tqdm
    ) -> None:
        """Update the progress bar with the latest metrics.

        Args:
            pbar: The progress bar to update.
        """

        pbar.set_postfix(
            loss=self.compute_loss(),
            accuracy=self.compute_mca(),
            compute_efficiency=self._get_compute_efficiency()
        )

    def update_loss(
            self,
            loss: float,
            batch_size: int
    ) -> None:
        """Update running and total loss.

        Args:
            loss: The loss for the mini-batch.
            batch_size: The number of samples in the mini-batch.
        """

        self.running_loss += loss * batch_size
        self.running_samples += batch_size
        self.total_loss += loss * batch_size
        self.total_samples += batch_size

    def update_mca(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor
    ) -> None:
        """Update running and total multiclass accuracy.

        Args:
            preds: The model predictions for the mini-batch.
            targets: The ground-truth labels for the mini-batch.
        """

        self.running_mca.update(preds, targets)
        self.total_mca.update(preds, targets)

    def compute_loss(
            self,
            total: bool = False
    ) -> float:
        """Compute the loss during training/validation.

        Args:
            total: A flag to indicate whether to compute the total loss
              or the running loss.

        Returns:
            The loss over a subset of mini-batches (total=False) or the
            entire dataset (total=True).
        """

        if total:
            return self.total_loss / self.total_samples
        return self.running_loss / self.running_samples

    def compute_mca(
            self,
            total: bool = False
    ) -> float:
        """Compute the multiclass accuracy during training/validation.

        Args:
            total: A flag to indicate whether to compute the total
              multiclass accuracy or the running multiclass accuracy.

        Returns:
            The multiclass accuracy over a subset of mini-batches
            (total=False) or the entire dataset (total=True).
        """

        if total:
            return self.total_mca.compute().item() * 100
        return self.running_mca.compute().item() * 100

    def log_scalars(self) -> None:
        """Log scalars to TensorBoard at specified batches.

        Log the loss and multiclass accuracy to TensorBoard at specified
        batch indices.  The batch indices are computed based on the
        total number of samples in the dataset and the desired number of
        per-epoch updates specified by ``self.tb_updates``.
        """

        if self.batch in self.tb_indices:
            # Compute global step
            dataloader = self.train_loader if self.is_training else self.val_loader
            global_step = (self.epoch - 1) * len(dataloader) + self.batch

            # Log scalars for consumption by TensorBoard
            self.writer.add_scalar(self.tb_tags["loss"], self.compute_loss(), global_step)
            self.writer.add_scalar(self.tb_tags["acc"], self.compute_mca(), global_step)

            # Reset loss and multiclass accuracy
            self._reset_metrics()

    def take_time(
            self,
            stage: str
    ) -> None:
        """Record timestamp.

        Args:
            stage: The stage of processing the mini-batch, either
              "start", "prep", or "proc".
        """

        timestamp = time.time()
        if stage == "start":
            self.start_time = timestamp
        elif stage == "prep":
            self.prep_time = timestamp
        elif stage == "proc":
            self.proc_time = timestamp

    def increment_batch(self) -> None:
        """Increment the batch number."""

        self.batch += 1

    def increment_epoch(self) -> None:
        """Increment the epoch number."""

        self.epoch += 1

    def flush_writer(self) -> None:
        """Flush the SummaryWriter instance."""

        self.writer.flush()

    def close_writer(self) -> None:
        """Close the SummaryWriter instance."""

        self.writer.close()

    def _set_tb_indices(self) -> None:
        """Set indices at which to log data to TensorBoard.

        Compute indices at which data is to be logged to TensorBoard.
        The indices are computed based on the total number of samples in
        the dataset and the desired number of per-epoch updates
        specified by ``self.tb_updates``.
        """

        dataloader = self.train_loader if self.is_training else self.val_loader
        total_samples = len(dataloader.dataset)
        sample_indices = torch.linspace(
            0, total_samples, self.tb_updates + 1
        )
        self.tb_indices = (
            torch.ceil(sample_indices / dataloader.batch_size)
        ).int().tolist()[1:]

    def _get_compute_efficiency(self) -> float:
        """Determine the compute efficiency during training/validation.

        Compute efficiency is defined as the percentage of time spent on
        processing the data relative to the total time spent on
        processing and preparing the data.  This metric is useful for
        identifying bottlenecks in the training loop related to data
        loading.

        Returns:
            The compute efficiency in percent.
        """

        prep_duration = self.prep_time - self.start_time
        proc_duration = self.proc_time - self.prep_time
        total_duration = prep_duration + proc_duration
        return proc_duration / total_duration * 100

    def _reset_metrics(
            self,
            total: bool = False
    ) -> None:
        """Reset the running or total metrics.

        Args:
            total: A flag to indicate whether to reset the total metrics
              or the running metrics.
        """

        if total:
            self.total_loss = 0.
            self.total_samples = 0
            self.total_mca.reset()
        else:
            self.running_loss = 0.
            self.running_samples = 0
            self.running_mca.reset()
