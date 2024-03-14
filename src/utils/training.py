import torch


def train_model(model, train_loader, val_loader, loss_fn, optimizer, cfg):
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

    for _ in range(cfg.params.num_epochs):
        # Train and validate model
        train_logs = run_epoch(model, train_loader, loss_fn, cfg, optimizer)
        val_logs = run_epoch(model, val_loader, loss_fn, cfg)

        # Log results
        logs["train"]["global_step"].extend(train_logs["global_step"])
        logs["train"]["loss"].extend(train_logs["loss"])
        logs["train"]["accuracy"].extend(train_logs["accuracy"])
        logs["val"]["global_step"].extend(val_logs["global_step"])
        logs["val"]["loss"].extend(val_logs["loss"])
        logs["val"]["accuracy"].extend(val_logs["accuracy"])

        # Increment epoch index
        cfg.logging.epoch_index += 1

    return logs


def run_epoch(model, dataloader, loss_fn, cfg, optimizer=None):
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
    model.train(is_training)

    # Determine batch indices at which to log to TensorBoard
    num_batches = len(dataloader)
    log_indices = torch.linspace(
        0, num_batches - 1, cfg.logging.intra_epoch_updates + 1
    ).int().tolist()
    if cfg.logging.epoch_index != 0:
        log_indices = log_indices[1:]

    # Disable gradients during evaluation
    with (torch.set_grad_enabled(is_training)):
        for batch_index, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(cfg.training.device)
            targets = targets.to(cfg.training.device)

            # Keep track of the number of samples
            samples = len(targets)
            running_samples += samples

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Accumulate loss
            running_loss += loss.item() * samples

            # Compute accuracy
            _, predictions = torch.max(outputs, 1)
            correct = (predictions == targets).sum().item()
            running_correct += correct

            # Backward pass and optimization
            if is_training:
                optimizer.zero_grad()  # zero gradients
                loss.backward()        # compute gradients
                optimizer.step()       # update weights

            # Log batch loss and accuracy to TensorBoard
            if batch_index in log_indices:
                # Compute global step
                global_step = cfg.logging.epoch_index * num_batches + batch_index + 1

                # Compute average loss and average accuracy per batch
                avg_batch_loss = running_loss / running_samples
                avg_batch_accuracy = (running_correct / running_samples) * 100  # in pct

                # Log to TensorBoard
                tag_loss = f"Loss/{'Train' if is_training else 'Val'}"
                tag_accuracy = f"Accuracy/{'Train' if is_training else 'Val'}"
                cfg.logging.writer.add_scalar(tag_loss, avg_batch_loss, global_step)
                cfg.logging.writer.add_scalar(tag_accuracy, avg_batch_accuracy, global_step)

                # Log to dictionary
                logs["global_step"].append(global_step)
                logs["loss"].append(avg_batch_loss)
                logs["accuracy"].append(avg_batch_accuracy)

                # Reset running totals
                running_samples = 0
                running_loss = 0.
                running_correct = 0

    # Flush writer after epoch for live updates
    cfg.logging.writer.flush()

    return logs
