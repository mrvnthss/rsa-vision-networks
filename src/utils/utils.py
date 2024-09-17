"""Utility functions used throughout this project.

Functions:
    * evaluate_classifier(model, test_loader, ...): Evaluate a
        classification model.
    * get_training_durations(log_file_path, drop_run_id=True): Parse a
        single log file to determine training durations.
    * get_training_results(log_file_path, mode, drop_run_id=True):
        Parse a single log file to determine training results.
    * get_upper_tri_matrix(sq_matrix): Extract the upper triangular part
        of a square matrix.
    * is_vector(x): Check if the input is a ``torch.Tensor`` of
        dimension 1.
    * parse_log_dir(log_dir, parse_fn, ...): Parse a (parent) directory
        of log files.
    * parse_tb_data(log_dir, extract_hparams=True, drop_run_id=True):
        Parse data stored in TensorBoard event files.
    * set_seeds(seed, cudnn_deterministic, cudnn_benchmark): Set random
        seeds for reproducibility.
"""


__all__ = [
    "evaluate_classifier",
    "get_training_durations",
    "get_training_results",
    "get_upper_tri_matrix",
    "is_vector",
    "parse_log_dir",
    "parse_tb_data",
    "set_seeds"
]

import inspect
import os
import random
import re
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from tbparse import SummaryReader
from torch import nn
from torchmetrics import MetricCollection
from tqdm import tqdm


def evaluate_classifier(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        metrics: MetricCollection,
        device: torch.device
) -> Dict[str, float]:
    """Evaluate a classification model.

    Args:
        model: The classification model to evaluate.
        test_loader: The dataloader providing test samples.
        criterion: The criterion to use for evaluation.
        metrics: The metrics to evaluate the model with.
        device: The device to perform evaluation on.

    Returns:
        The loss along with the computed metrics, evaluated on the test
        set.
    """

    model.eval()
    metrics.reset()
    metrics.to(device)

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
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Make predictions and update metrics
            predictions = model(inputs)
            metrics.update(predictions, targets)

            # Compute loss and accumulate
            loss = criterion(predictions, targets)
            samples = targets.size(dim=0)
            running_loss += loss.item() * samples
            running_samples += samples

    metric_values = metrics.compute()
    results = {
        "Loss": running_loss / running_samples,
        **metric_values
    }

    return results


def get_training_durations(
        log_file_path: str,
        drop_run_id: Optional[bool] = True
) -> pd.DataFrame:
    """Parse a single log file to determine training durations.

    Note:
        This functon assumes that the log file to parse corresponds to a
        single Hydra run.  This run can, however, include multiple
        training runs (e.g., corresponding to folds in a
        cross-validation setup).

    Args:
        log_file_path: The full path to the log file.
        drop_run_id: Whether to drop the run ID from the DataFrame if
          the log file only contains information about a single run.

    Returns:
        A DataFrame containing the duration of each epoch of all runs
        along with the hyperparameters used in these runs.
    """

    # Extract individual training runs from log file
    training_runs = _extract_training_runs(log_file_path)

    # Define patterns to match against
    epoch_pattern = r"EPOCH \[\d+/\d+\]"
    timestamp_pattern = r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]"

    data = []
    for run_id, run_lines in training_runs.items():
        # Start of training
        timestamp = datetime.strptime(
            re.match(timestamp_pattern, run_lines[0]).group(0), "[%Y-%m-%d %H:%M:%S,%f]"
        )
        data.append({"run_id": run_id + 1, "timestamp": timestamp})

        # Individual epochs
        for line in run_lines[1:]:
            match = re.search(epoch_pattern, line)
            if match:
                timestamp = datetime.strptime(
                    re.match(timestamp_pattern, line).group(0), "[%Y-%m-%d %H:%M:%S,%f]"
                )
                data.append({"run_id": run_id + 1, "timestamp": timestamp})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Compute duration (in secs) of each epoch as difference between timestamps for each run
    df["duration"] = df.groupby("run_id")["timestamp"].diff().dt.total_seconds()
    df = df.dropna()[["run_id", "duration"]].reset_index(drop=True)

    # Index epochs per run
    df["epoch"] = df.groupby("run_id").cumcount() + 1

    # Add hyperparameter settings to DataFrame
    hparams = _extract_hparams(_extract_run_dir(log_file_path))
    for k, v in hparams.items():
        df[k] = v

    # Reorder columns and possibly drop run ID, if requested
    df = df[["run_id", *hparams.keys(), "epoch", "duration"]]
    if len(training_runs) == 1 and drop_run_id:
        df = df.drop(columns="run_id")

    return df


def get_training_results(
        log_file_path: str,
        mode: Literal["Train", "Val"],
        drop_run_id: Optional[bool] = True
) -> pd.DataFrame:
    """Parse a single log file to determine training results.

    Note:
        This functon assumes that the log file to parse corresponds to a
        single Hydra run.  This run can, however, include multiple
        training runs (e.g., corresponding to folds in a
        cross-validation setup).

    Args:
        log_file_path: The full path to the log file.
        mode: Whether to extract results for the training or validation
          set.
        drop_run_id: Whether to drop the run ID from the DataFrame if
          the log file only contains information about a single run.

    Returns:
        A DataFrame containing the results of the best performing model
        for each run along with the hyperparameters used in these runs.
    """

    # Extract individual training runs from log file
    training_runs = _extract_training_runs(log_file_path)

    # Define patterns to match against
    epoch_pattern = r"EPOCH \[(\d+)/\d+\]\s+TRAIN: (.+?)\s+VAL: (.+)"
    results_pattern = (
        r"Best performing model achieved a score of ([\d.]+) \((.+)\) on the "
        r"(training|validation) set after (\d+) epochs of training."
    )

    data = []
    for run_id, run_lines in training_runs.items():
        # Get number of epochs trained (best performing model)
        run_dict = {
            "run_id": run_id + 1,
            "Epochs": int(re.search(results_pattern, run_lines[-1]).group(4))
        }

        # Extract training results
        for line in run_lines:
            match = re.search(epoch_pattern, line)
            if match:
                epoch = int(match.group(1))
                if epoch == run_dict["Epochs"]:
                    metrics = match.group(2 if mode == "Train" else 3).split("  ")
                    for metric in metrics:
                        k, v = metric.split(": ")
                        run_dict[k] = float(v)

        data.append(run_dict)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Add hyperparameter settings to DataFrame
    hparams = _extract_hparams(_extract_run_dir(log_file_path))
    for k, v in hparams.items():
        df[k] = v

    # Reorder columns of DataFrame
    hparam_names = list(hparams.keys())
    hparam_names.append("run_id")
    remaining_columns = [col for col in df.columns if col not in hparam_names]
    df = df[[*hparam_names, *remaining_columns]]

    # Drop run ID, if requested
    if len(training_runs) == 1 and drop_run_id:
        df = df.drop(columns="run_id")

    return df


def get_upper_tri_matrix(sq_matrix: torch.Tensor) -> torch.Tensor:
    """Extract the upper triangular part of a square matrix.

    Args:
        sq_matrix: The square matrix from which to extract the upper
          triangular matrix (excluding the diagonal).

    Returns:
        The upper triangular matrix (excluding the diagonal) of the
        square matrix ``sq_matrix``, flattened into a vector using
        row-major order.
    """

    mask = torch.triu(torch.ones_like(sq_matrix, dtype=torch.bool), diagonal=1)
    return sq_matrix[mask]


def is_vector(x: torch.Tensor) -> bool:
    """Check if the input is a ``torch.Tensor`` of dimension 1."""

    return isinstance(x, torch.Tensor) and x.dim() == 1


def parse_log_dir(
        log_dir: str,
        parse_fn: Callable,
        mode: Optional[Literal["Train", "Val"]] = None,
        drop_run_id: Optional[bool] = True
) -> pd.DataFrame:
    """Parse a (parent) directory of log files.

    Note:
        The ``parse_fn`` can be one of the following functions:
          * get_training_durations(log_file_path, drop_run_id=False)
          * get_training_results(log_file_path, mode, drop_run_id=False)

    Args:
        log_dir: The path of the (parent) directory storing the log
          files.
        parse_fn: The function to use for parsing individual log files.
        mode: Whether to extract results for the training or validation
          set from the individual log files.  Should only be provided
          when ``parse_fn`` takes an additional ``mode`` argument.
        drop_run_id: Whether to drop the run ID from the DataFrame
          returned by the ``parse_fn`` if the log file being parsed only
          contains information about a single run.

    Returns:
        A DataFrame containing the parsed data from the log files.

    Raises:
        ValueError: If no log files are found in the directory.
    """

    log_file_path = None
    combined_data = []
    hparam_names = set()

    if mode is not None and "mode" in inspect.signature(parse_fn).parameters:
        parse_fn = partial(parse_fn, mode=mode, drop_run_id=drop_run_id)
    else:
        parse_fn = partial(parse_fn, drop_run_id=drop_run_id)

    # Parse individual log files
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".log"):
                log_file_path = os.path.join(root, file)
                hparam_names.update(_extract_hparams(_extract_run_dir(log_file_path)).keys())
                df = parse_fn(log_file_path=log_file_path)
                combined_data.append(df)

    if log_file_path is None:
        raise ValueError(f"No log files found in directory: {log_dir}")

    # Combine into single DataFrame
    # NOTE: The ``concat`` function handles non-overlapping columns by filling in NaNs
    combined_df = pd.concat(
        combined_data,
        join="outer",
        ignore_index=True
    )

    # Reorder columns of DataFrame
    hparam_names = list(hparam_names)
    hparam_names.sort()
    hparam_names.extend([col for col in ["run_id", "epoch"] if col in combined_df.columns])
    remaining_columns = [col for col in combined_df.columns if col not in hparam_names]
    combined_df = combined_df[[*hparam_names, *remaining_columns]]

    # Sort DataFrame and reset index
    combined_df = combined_df.sort_values(by=hparam_names)
    combined_df.reset_index(drop=True, inplace=True)

    return combined_df


def parse_tb_data(
        log_dir: str,
        extract_hparams: bool = True,
        drop_run_id: Optional[bool] = True
) -> pd.DataFrame:
    """Parse data stored in TensorBoard event files.

    Note:
        When ``extract_hparams`` is set to True, the ``log_dir`` must
        be the immediate parent directory of the subdirectories
        specifying the hyperparameters.

    Args:
        log_dir: The path of the (parent) directory storing the
          TensorBoard event files.
        extract_hparams: Whether to extract hyperparameters from the
          names of the subdirectories specifying the run configurations.
        drop_run_id: Whether to drop the run ID from the DataFrame if
          there is only a single run per hyperparameter configuration.

    Returns:
        A DataFrame containing the training data in long format.
    """

    # Helper function to extract run ID from directory names
    def _extract_run_id(dir_name: str) -> int:
        match = re.search(r"/run(\d+)", dir_name)
        return int(match.group(1)) if match else 1

    # Read in data in long format, and add "global_id"
    df = SummaryReader(log_dir, extra_columns={"dir_name"}).scalars
    df["global_id"] = df.groupby("dir_name").ngroup() + 1

    # Add "run_id" distinguishing individual runs for the same hyperparameter configuration
    df["run_id"] = df["dir_name"].apply(_extract_run_id)

    # Extract name of metric and mode of training (training vs. validation)
    df[["metric", "mode"]] = df["tag"].str.split("/", expand=True)

    # Extract hyperparameters from subdirectories and add to DataFrame
    if extract_hparams:
        # NOTE: Here we assume that the ``log_dir`` is the immediate parent directory of the
        #       subdirectories specifying the hyperparameter configurations.
        subdir_names = df["dir_name"].str.split("/", expand=True)[0]
        hparams = subdir_names.apply(_extract_hparams)
        df = pd.concat([df, pd.DataFrame(hparams.tolist())], axis=1)
        hparam_names = hparams.apply(lambda x: list(x.keys())).explode().unique()

    # Reorder columns
    if extract_hparams:
        col_order = ["global_id", "run_id", *hparam_names, "mode", "metric", "step", "value"]
    else:
        col_order = ["global_id", "run_id", "mode", "metric", "step", "value"]
    df = df[col_order]

    # Sort DataFrame
    df = df.sort_values(by=["global_id", "run_id", "mode", "metric", "step"], ascending=True)
    df.reset_index(drop=True, inplace=True)

    if drop_run_id and df["run_id"].nunique() == 1:
        df = df.drop(columns="run_id")

    return df


def set_seeds(
        seed: int,
        cudnn_deterministic: bool,
        cudnn_benchmark: bool
) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: The random seed to use.
        cudnn_deterministic: Whether to enforce deterministic behavior
          of cuDNN.
        cudnn_benchmark: Whether to enable cuDNN benchmark mode.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def _extract_hparams(run_dir: str) -> Dict[str, Union[int, float, str]]:
    """Extract hyperparameters from a directory name.

    Args:
        run_dir: The name of the directory specifying the hyperparameter
          configuration of a run (i.e., not the full path).

    Returns:
        A dictionary containing information about the hyperparameters
        used in the run.
    """

    params_dict = {}
    for param in run_dir.split(","):
        k, v = param.split("=")
        # Convert to numeric, if possible
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        params_dict[k] = v
    return params_dict


def _extract_run_dir(log_file_path: str) -> str:
    """Extract the name of the directory specifying a run configuration.

    Args:
        log_file_path: The full path to the log file.

    Returns:
        The name of the directory (not the full path) specifying the run
        configuration.

    Raises:
        ValueError: If no run directory can be extracted from the path
          to the log file.
    """

    match = re.search(
        r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}/([^/]+)/logs",
        log_file_path
    )

    if not match:
        raise ValueError(f"Could not extract directory name from path: {log_file_path}")

    return match.group(1)


def _extract_training_runs(log_file_path: str) -> Dict[int, List[str]]:
    """Extract individual training runs from a log file.

    Args:
        log_file_path: The full path to the log file.

    Returns:
        A dictionary mapping run indices to a list of lines of the log
        file corresponding to that run.
    """

    # Read in lines of log file
    with open(log_file_path, "r", encoding="utf-8") as file:
        log_lines = file.readlines()

    # Get indices of lines corresponding to starts of individual training loops
    start_pattern = r"Starting training loop \.\.\."
    start_indices = [
        idx for idx, line in enumerate(log_lines)
        if re.search(start_pattern, line)
    ]

    # Get indices of lines corresponding to ends of individual training loops
    end_pattern = (
        r"Best performing model achieved a score of ([\d.]+) \((.+)\) on the "
        r"(training|validation) set after (\d+) epochs of training."
    )
    end_indices = [
        idx for idx, line in enumerate(log_lines)
        if re.search(end_pattern, line)
    ]

    # Split log lines into individual training runs
    training_runs = {}
    for run_idx, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
        single_run = log_lines[start_idx:end_idx + 1]
        training_runs[run_idx] = single_run

    return training_runs
