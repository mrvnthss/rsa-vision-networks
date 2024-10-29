"""Utility functions related to data handling.

Functions:
    * get_training_durations(log_file_path, drop_run_id, ...): Parse a
        single log file to determine training durations.
    * get_training_results(log_file_path, mode, ...): Parse a single log
        file to determine training results.
    * parse_log_dir(log_dir, parse_fn, ...): Parse a (parent) directory
        of log files.
    * parse_tb_data(log_dir, extract_hparams=True, drop_run_id=True):
        Parse data stored in TensorBoard event files.
"""


__all__ = [
    "get_training_durations",
    "get_training_results",
    "parse_log_dir",
    "parse_tb_data"
]

import inspect
import os
import re
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from tbparse import SummaryReader


def get_training_durations(
        log_file_path: str,
        drop_run_id: bool = True,
        extract_hparams: bool = False
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
        extract_hparams: Whether to extract hyperparameters from the
          full path to the log file.

    Returns:
        A DataFrame containing the duration of each epoch of all runs
        along with the hyperparameters used in these runs.
    """

    # Extract individual training runs from log file
    training_runs = _extract_training_runs(log_file_path)
    if not training_runs:
        return pd.DataFrame()

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

    # Extract hyperparameters from directory and add to DataFrame, reorder columns
    if extract_hparams:
        hparams = _extract_hparams(_extract_run_dir(log_file_path))
        for k, v in hparams.items():
            df[k] = v
        col_order = [*hparams, "run_id", "epoch", "duration"]
    else:
        col_order = ["run_id", "epoch", "duration"]
    df = df[col_order]

    # Drop run ID, if requested
    if len(training_runs) == 1 and drop_run_id:
        df = df.drop(columns="run_id")

    return df


def get_training_results(
        log_file_path: str,
        mode: Literal["train", "val"],
        drop_run_id: bool = True,
        extract_hparams: bool = False
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
        extract_hparams: Whether to extract hyperparameters from the
          full path to the log file.

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
                    metrics = match.group(2 if mode == "train" else 3).split("  ")
                    for metric in metrics:
                        k, v = metric.split(": ")
                        run_dict[k] = float(v)

        data.append(run_dict)

    # Handle the case in which no results could be extracted from log file
    if not data:
        data = [{
            "run_id": np.nan,
            "Epochs": np.nan
        }]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Extract hyperparameters from directory and add to DataFrame, set column order
    if extract_hparams:
        hparams = _extract_hparams(_extract_run_dir(log_file_path))
        for k, v in hparams.items():
            df[k] = v
        hparam_names = list(hparams.keys()) + ["run_id"]
        remaining_columns = [col for col in df.columns if col not in hparam_names]
        col_order = [*hparam_names, *remaining_columns]
    else:
        col_order = ["run_id", *[col for col in df.columns if col != "run_id"]]

    # Move "Epochs" to the very end, reorder columns
    col_order.remove("Epochs")
    col_order.append("Epochs")
    df = df[col_order]

    # Drop run ID, if requested
    if len(training_runs) == 1 and drop_run_id:
        df = df.drop(columns="run_id")

    return df


def parse_log_dir(
        log_dir: str,
        parse_fn: Callable,
        mode: Optional[Literal["train", "val"]] = None,
        drop_run_id: bool = True
) -> pd.DataFrame:
    """Parse a (parent) directory of log files.

    Note:
        The ``parse_fn`` can be one of the following functions:
          * get_training_durations
          * get_training_results

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
        parse_fn = partial(
            parse_fn,
            mode=mode,
            drop_run_id=drop_run_id,
            extract_hparams=True
        )
    else:
        parse_fn = partial(
            parse_fn,
            drop_run_id=drop_run_id,
            extract_hparams=True
        )

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

    # Extract hyperparameters from subdirectories and add to DataFrame, reorder columns
    if extract_hparams:
        # NOTE: Here we assume that the ``log_dir`` is the immediate parent directory of the
        #       subdirectories specifying the hyperparameter configurations.
        subdir_names = df["dir_name"].str.split("/", expand=True)[0]
        hparams = subdir_names.apply(_extract_hparams)
        df = pd.concat([df, pd.DataFrame(hparams.tolist())], axis=1)
        hparam_names = hparams.apply(lambda x: list(x.keys())).explode().unique()
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
    for subdir in run_dir.split("/"):
        for param in subdir.split(","):
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
        r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}/(.+)/logs",
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
