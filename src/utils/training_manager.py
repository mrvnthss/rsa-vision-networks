"""A class to handle auxiliary tasks during training and validation."""


import time

from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm


class TrainingManager:
    """A manager to handle auxiliary tasks during training/validation.

    Handles auxiliary tasks during training and validation, such as
    switching between training and validation modes, updating the
    progress bar, logging metrics to TensorBoard, and computing the
    compute efficiency.

    Params:
        model: The model to be trained.
        train_loader: The dataloader providing training samples.
        val_loader: The dataloader providing validation samples.
        device: The device to train on.
        cfg: The training configuration.

    (Additional) Attributes:
        mca: A multiclass accuracy metric to track model performance.
        writer: A SummaryWriter instance to log metrics to TensorBoard.
        log_indices: Indices at which to log metrics to TensorBoard.
        tb_tags: Tags for logging metrics to TensorBoard.
        is_training: A flag to indicate whether the model is training.
        start_epoch: The starting epoch number.
        epoch: The current epoch number.
        batch: The current batch number.
        samples: The number of samples processed within one epoch.
        running_loss: The running loss during training/validation.
        start_time: A timestamp indicating the start of processing a
          mini-batch.
        prep_time: A timestamp indicating the end of preparing a
          mini-batch.
        proc_time: A timestamp indicating the end of processing a
          mini-batch.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            device: torch.device,
            cfg: DictConfig
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg

        self.mca = MulticlassAccuracy(
            num_classes=cfg.dataset.num_classes,
            average="micro"
        ).to(self.device)

        self.writer = SummaryWriter(cfg.paths.logs)
        self.log_indices = []
        self.tb_tags = {}

        self.is_training = True

        self.start_epoch = 1
        self.epoch = 1
        self.batch = 1
        self.samples = 0
        self.running_loss = 0.

        self.start_time = 0.
        self.prep_time = 0.
        self.proc_time = 0.

    def prepare_run(self, state: str) -> None:
        """Perform setup before training or validation.

        Set the model to training or evaluation mode, reset the batch
        index, compute indices at which to log metrics to TensorBoard,
        and set TensorBoard tags.
        """
        self.is_training = state == "train"
        self.model.train(self.is_training)
        self.batch = 1
        self._set_log_indices()
        self.tb_tags = {
            "loss": f"loss/{'train' if self.is_training else 'val'}",
            "acc": f"acc/{'train' if self.is_training else 'val'}"
        }

    def visualize_model(self, inputs: torch.Tensor) -> None:
        self.writer.add_graph(self.model, inputs)

    def get_pbar(self) -> tqdm:
        """Decorate a dataloader with a ``tqdm`` progress bar."""
        dataloader = self.train_loader if self.is_training else self.val_loader
        final_epoch = self.start_epoch + self.cfg.training.num_epochs - 1
        desc = (f"Epoch [{self.epoch}/{final_epoch}]    "
                f"{'Train' if self.is_training else 'Val'}")
        return tqdm(dataloader, desc=desc, leave=False, unit="batch")

    def update_pbar(self, pbar: tqdm) -> None:
        pbar.set_postfix(
            loss=self._compute_loss(),
            accuracy=self._compute_mca(),
            compute_efficiency=self._get_compute_efficiency()
        )

    def update_loss(
            self,
            loss: float,
            batch_size: int
    ) -> None:
        self.running_loss += loss * batch_size
        self.samples += batch_size

    def update_mca(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor
    ) -> None:
        self.mca.update(preds, targets)

    def log_metrics(self) -> None:
        """Log metrics to TensorBoard at specified intervals."""
        if self.batch in self.log_indices:
            # Compute global step
            dataloader = self.train_loader if self.is_training else self.val_loader
            global_step = (self.epoch - 1) * len(dataloader) + self.batch

            # Log metrics to TensorBoard
            self.writer.add_scalar(self.tb_tags["loss"], self._compute_loss(), global_step)
            self.writer.add_scalar(self.tb_tags["acc"], self._compute_mca(), global_step)

            # Reset loss and multiclass accuracy
            self._reset_metrics()

    def take_time(self, stage: str) -> None:
        timestamp = time.time()
        if stage == "start":
            self.start_time = timestamp
        elif stage == "prep":
            self.prep_time = timestamp
        elif stage == "proc":
            self.proc_time = timestamp

    def increment_batch(self) -> None:
        self.batch += 1

    def increment_epoch(self) -> None:
        self.epoch += 1

    def flush_writer(self) -> None:
        self.writer.flush()

    def close_writer(self) -> None:
        self.writer.close()

    def _set_log_indices(self) -> None:
        """Set indices at which to log metrics to TensorBoard.

        Compute indices at which metrics are to be logged to
        TensorBoard.  The indices are computed based on the total number
        of samples in the dataset and the logging frequency specified in
        the training configuration ``self.cfg``.
        """
        dataloader = self.train_loader if self.is_training else self.val_loader
        total_samples = len(dataloader.dataset)
        sample_intervals = torch.linspace(
            0, total_samples, self.cfg.training.log_frequency + 1
        )
        self.log_indices = (
            torch.ceil(sample_intervals / dataloader.batch_size)
        ).int().tolist()[1:]

    def _get_compute_efficiency(self) -> float:
        """Determine the compute efficiency during training/validation.

        Compute efficiency is defined as the percentage of time spent on
        processing the data relative to the total time spent on
        processing and preparation.  This metric is useful for
        identifying bottlenecks in the training loop related to data
        loading.
        """
        prep_duration = self.prep_time - self.start_time
        proc_duration = self.proc_time - self.prep_time
        total_duration = prep_duration + proc_duration
        return proc_duration / total_duration * 100

    def _compute_loss(self) -> float:
        return self.running_loss / self.samples

    def _compute_mca(self) -> float:
        return self.mca.compute().item() * 100

    def _reset_metrics(self) -> None:
        self.running_loss = 0.
        self.samples = 0
        self.mca.reset()
