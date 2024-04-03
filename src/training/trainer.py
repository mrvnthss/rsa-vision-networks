import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            device,
            cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device
        self.writer = SummaryWriter(cfg.logging.tb_dir)

    def train(self):
        # Initialize dictionary to log results
        logs = {
            "train": {
                "global_step": [],
                "loss": [],
                "accuracy": []
            },
            "val": {
                "global_step": [],
                "loss": [],
                "accuracy": []
            }
        }

        for _ in range(self.cfg.training.num_epochs):
            # Train and validate model
            train_logs = self._run_epoch(self.train_loader, self.optimizer)
            val_logs = self._run_epoch(self.val_loader)

            # Log results
            logs["train"]["global_step"].extend(train_logs["global_step"])
            logs["train"]["loss"].extend(train_logs["loss"])
            logs["train"]["accuracy"].extend(train_logs["accuracy"])
            logs["val"]["global_step"].extend(val_logs["global_step"])
            logs["val"]["loss"].extend(val_logs["loss"])
            logs["val"]["accuracy"].extend(val_logs["accuracy"])

            # Increment epoch index
            self.cfg.logging.epoch_index += 1

        # Close TensorBoard writer and inform user of training completion
        self.writer.close()
        print("Training complete!")

        return logs

    def _run_epoch(
            self,
            dataloader,
            optimizer=None):
        # Initialize dictionary to log intra-epoch results
        logs = {
            "global_step": [],
            "loss": [],
            "accuracy": []
        }

        # Running totals to report progress to TensorBoard
        running_samples = 0
        running_loss = 0.
        running_correct = 0

        # Set training/evaluation mode
        is_training = optimizer is not None
        self.model.train(is_training)

        # Determine batch indices at which to log to TensorBoard
        num_batches = len(dataloader)
        log_indices = torch.linspace(
            0, num_batches - 1, self.cfg.logging.intra_epoch_updates + 1
        ).int().tolist()
        if self.cfg.logging.epoch_index != 0:
            log_indices = log_indices[1:]

        # Set tags for TensorBoard logging
        tag_loss = f"Loss/{'Train' if is_training else 'Val'}"
        tag_accuracy = f"Accuracy/{'Train' if is_training else 'Val'}"

        # Prepare progress bar
        desc = (f"Epoch [{self.cfg.logging.epoch_index + 1}/{self.cfg.training.num_epochs}]    "
                f"{'Train' if is_training else 'Val'}")
        pbar = tqdm(dataloader, desc=desc, leave=False, unit="batch")

        # Disable gradients during evaluation
        with (torch.set_grad_enabled(is_training)):
            for batch_index, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Keep track of the number of samples
                samples = len(targets)
                running_samples += samples

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                # Accumulate loss
                running_loss += loss.item() * samples

                # Compute accuracy
                _, predictions = torch.max(outputs, 1)
                correct = (predictions == targets).sum().item()
                running_correct += correct

                # Update progress bar
                avg_batch_loss = running_loss / running_samples
                avg_batch_accuracy = (running_correct / running_samples) * 100  # in pct
                pbar.set_postfix(
                    loss=avg_batch_loss,
                    accuracy=avg_batch_accuracy
                )

                # Backward pass and optimization
                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Log batch loss and accuracy
                if batch_index in log_indices:
                    # Log to TensorBoard
                    global_step = self.cfg.logging.epoch_index * num_batches + batch_index + 1
                    self.writer.add_scalar(tag_loss, avg_batch_loss, global_step)
                    self.writer.add_scalar(tag_accuracy, avg_batch_accuracy, global_step)

                    # Log to dictionary
                    logs["global_step"].append(global_step)
                    logs["loss"].append(avg_batch_loss)
                    logs["accuracy"].append(avg_batch_accuracy)

                    # Reset running totals
                    running_samples = 0
                    running_loss = 0.
                    running_correct = 0

        # Flush writer after epoch for live updates
        self.writer.flush()

        return logs
