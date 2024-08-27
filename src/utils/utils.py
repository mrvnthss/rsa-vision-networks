"""Utility functions used throughout this project.

Functions:
    * evaluate_classifier: Evaluate a classification model.
    * get_training_durations: Parse a single log file to determine
        training durations.
    * get_training_results: Parse a single log file to determine
        training results.
    * parse_log_dir: Parse a directory of log files.
    * preprocess_training_data: Preprocess training data in the form of
        TensorBoard event files.
"""


__all__ = [
    "evaluate_classifier",
    "get_training_durations",
    "get_training_results",
    "parse_log_dir",
    "preprocess_training_data"
]

import os
import re
from datetime import datetime
from functools import partial
from typing import Callable, Dict, Literal, Optional, Union

import pandas as pd
import torch
from tbparse import SummaryReader
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm


def evaluate_classifier(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> Dict[str, float]:
    """Evaluate a classification model.

    Args:
        model: The classification model to evaluate.
        test_loader: The dataloader providing test samples.
        criterion: The criterion to use for evaluation.
        device: The device to perform evaluation on.

    Returns:
        The classification accuracy (top-1 and top-5) and the loss
        evaluated on the test set.
    """

    model.eval()

    acc_1 = MulticlassAccuracy(
        num_classes=len(test_loader.dataset.classes),
        top_k=1,
        average="micro",
        multidim_average="global"
    ).to(device)

    acc_5 = MulticlassAccuracy(
        num_classes=len(test_loader.dataset.classes),
        top_k=5,
        average="micro",
        multidim_average="global"
    ).to(device)

    running_loss = 0.
    running_samples = 0

    # Set up progress bar
    pbar = tqdm(
        test_loader,
        desc=f"Evaluating {model.__class__.__name__}",
        total=len(test_loader),
        leave=True,
        unit="batch"
    )

    with torch.no_grad():
        for features, targets in pbar:
            features, targets = features.to(device), targets.to(device)

            # Make predictions
            predictions = model(features)

            # Track multiclass accuracy
            acc_1.update(predictions, targets)
            acc_5.update(predictions, targets)

            # Compute loss and accumulate
            loss = criterion(predictions, targets)
            samples = targets.size(dim=0)
            running_loss += loss.item() * samples
            running_samples += samples

    results = {
        "loss": running_loss / running_samples,
        "acc@1": acc_1.compute().item(),
        "acc@5": acc_5.compute().item()
    }

    return results


def get_training_durations(log_file: str) -> pd.DataFrame:
    """Parse a single log file to determine training durations.

    Args:
        log_file: The (full) path to the log file.

    Returns:
        A pandas DataFrame containing the duration of each epoch of a
        single run.  The DataFrame also contains the hyperparameters
        associated with the run.
    """

    start_pattern = r"Starting training loop \.\.\."
    epoch_pattern = r"EPOCH \[\d+/\d+\]"
    timestamp_pattern = r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]"

    data = []
    with open(log_file, "r", encoding="utf-8") as file:
        for line in file:
            start_match = re.search(start_pattern, line)
            epoch_match = re.search(epoch_pattern, line)

            if start_match or epoch_match:
                timestamp = datetime.strptime(
                    re.match(timestamp_pattern, line).group(0),
                    "[%Y-%m-%d %H:%M:%S,%f]"
                )
                data.append(timestamp)

    df = pd.DataFrame(data, columns=["timestamp"])

    # Compute duration (in secs) of each epoch as difference between timestamps
    df = df.diff()[1:]
    df["timestamp"] = df["timestamp"].dt.total_seconds()
    df = df.rename(columns={"timestamp": "duration"})

    # Add meta-data
    run_params = _extract_params(_extract_run_dir(log_file))
    for key, value in run_params.items():
        df[key] = value
    df["epoch"] = range(1, len(df) + 1)

    # Reorder columns
    df = df[[*run_params.keys(), "epoch", "duration"]]

    return df


def get_training_results(
        log_file: str,
        mode: Literal["Train", "Val"]
) -> pd.DataFrame:
    """Parse a single log file to determine training results.

    Args:
        log_file: The (full) path to the log file.
        mode: Whether to extract results for the training or validation
          set.

    Returns:
        A DataFrame containing the hyperparameters and training results
        of a single run.
    """

    epoch_pattern = r"EPOCH \[(\d+)/\d+\]\s+TRAIN: (.+?)\s+VAL: (.+)"
    end_of_training_pattern = (
        r"Best performing model achieved a score of ([\d.]+) \((.+)\) on the "
        r"(training|validation) set after (\d+) epochs of training."
    )

    # Extract hyperparameters and store in dictionary
    results = {**_extract_params(_extract_run_dir(log_file))}

    # Extract number of epochs trained (best performing model)
    with open(log_file, "r", encoding="utf-8") as file:
        for line in file:
            match = re.search(end_of_training_pattern, line)
            if match:
                results["Epochs"] = int(match.group(4))
                break

    # Extract training results
    with open(log_file, "r", encoding="utf-8") as file:
        for line in file:
            match = re.search(epoch_pattern, line)
            if match:
                epoch = int(match.group(1))
                if epoch == results["Epochs"]:
                    metrics = match.group(2 if mode == "Train" else 3).split("  ")
                    for metric in metrics:
                        key, value = metric.split(": ")
                        results[key] = float(value)

    return pd.DataFrame(results, index=[0])


def parse_log_dir(
        log_dir: str,
        parse_fn: Callable,
        mode: Optional[Literal["Train", "Val"]] = None
) -> pd.DataFrame:
    """Parse a directory of log files.

    Args:
        log_dir: The path of the directory storing the log files.
        parse_fn: The function to use for parsing the log files.
        mode: Whether to extract results for the training or validation
          set.  Should only be provided when ``parse_fn`` takes an
          additional ``mode`` argument.

    Returns:
        A DataFrame containing the extracted results for each run.

    Raises:
        ValueError: If no log files are found in the directory.
    """

    log_file = None
    combined_data = []

    if mode is not None:
        parse_fn = partial(parse_fn, mode=mode)

    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".log"):
                log_file = os.path.join(root, file)
                df = parse_fn(log_file=log_file)
                combined_data.append(df)

    if log_file is None:
        raise ValueError(f"No log files found in directory: {log_dir}")

    run_params = _extract_params(_extract_run_dir(log_file))

    # Combine into single dataframe and sort values
    combined_df = pd.concat(combined_data, ignore_index=True)
    if "epoch" in combined_df.columns:
        combined_df = combined_df.sort_values(by=[*run_params, "epoch"])
    else:
        combined_df = combined_df.sort_values(by=[*run_params])
    combined_df.reset_index(drop=True, inplace=True)

    return combined_df


def preprocess_training_data(log_dir: str) -> pd.DataFrame:
    """Preprocess training data in the form of TensorBoard event files.

    Args:
        log_dir: The path of the directory storing the TensorBoard
           event files.

    Returns:
        A DataFrame containing the training data in long format.
    """

    # Read in data in long format
    df = SummaryReader(log_dir, extra_columns={"dir_name"}).scalars

    # Add run_id
    df["run_id"] = df.groupby("dir_name").ngroup()

    # Extract name of metric and mode of training (training vs. validation)
    df[["metric", "mode"]] = df["tag"].str.split("/", expand=True)

    # Extract parameters from directory name
    params = df["dir_name"].str.rstrip("/logs").apply(_extract_params)
    df = pd.concat([df, pd.DataFrame(params.tolist())], axis=1)
    unique_params = list(params[0].keys())

    return df[["run_id", *unique_params, "mode", "metric", "step", "value"]]


def _extract_params(run_dir: str) -> Dict[str, Union[int, float, str]]:
    """Extract hyperparameters from a run directory.

    Note:
        The ``run_dir`` is not the full path, but the directory name
        containing the training logs for a single run.

    Args:
        run_dir: The directory storing training logs corresponding to a
          specific hyperparameter configuration (i.e., a single run).

    Returns:
        A dictionary containing hyperparameters and their values.
    """

    params_dict = {}
    for param in run_dir.split(","):
        tag, value = param.split("=")
        # Convert parameter value to numeric, if possible
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        params_dict[tag] = value
    return params_dict


def _extract_run_dir(log_file: str) -> str:
    """Extract the run directory from a log file path.

    Args:
        log_file: The (full) path to the log file.

    Returns:
        The directory name (not the full path) containing the training
        logs of a single run.

    Raises:
        ValueError: If the run directory cannot be extracted from the
          log file.
    """

    match = re.search(
        r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}/([^/]+)/logs",
        log_file
    )

    if not match:
        raise ValueError(f"Could not extract run directory from log file: {log_file}")

    return match.group(1)
