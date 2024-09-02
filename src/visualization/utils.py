"""Utility functions for general visualization purposes.

Functions:
    * save_figure(fig, f_path, dpi=300): Save a matplotlib figure.
    * smooth_ts(raw_ts, weight): Smooth a time series using EMA.
"""


import math
import warnings
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from typing_extensions import Union


def save_figure(
        fig: Figure,
        f_path: Union[str, Path],
        dpi: int = 300
) -> None:
    """Save a matplotlib figure.

    Args:
        fig: The matplotlib figure object to be saved.
        f_path: The file path where the figure should be saved.
        dpi: The resolution of the saved figure.
    """

    f_path = Path(f_path)
    f_path.parent.mkdir(parents=True, exist_ok=True)

    if f_path.exists():
        warnings.warn(f"File {f_path} already exists! Figure not saved.")
    else:
        fig.savefig(f_path, bbox_inches="tight", dpi=dpi)
        print(f"Figure saved successfully as {f_path}.")


def smooth_ts(
        raw_ts: pd.Series,
        weight: float
) -> pd.Series:
    """Smooth a time series using EMA.

    Adapted from https://github.com/tensorflow/tensorboard/blob/master/tensorboard/components/vz_line_chart2/line-chart.ts.

    Args:
        raw_ts: The raw time series to be smoothed.
        weight: The weight to be used in the exponential moving average.
          Must be in the range (0, 1).

    Returns:
        The smoothed time series.

    Raises:
        ValueError: If ``weight`` is not in the range (0, 1).
    """

    # Force weight to be in the range (0, 1)
    if weight <= 0 or weight >= 1:
        raise ValueError(
            f"'weight' should be between 0 and 1 (both exclusive), but got {weight}."
        )

    s_t = 0  # moving average at time t (not de-biased)
    num_accum = 0  # number of values accumulated (for de-biasing purposes)

    ts_name, ts_index = raw_ts.name, raw_ts.index
    ts_name = f"{ts_name}_ema"
    raw_ts = raw_ts.to_list()
    smoothed_ts = []

    for x_t in raw_ts:
        s_t = s_t * weight + (1 - weight) * x_t
        num_accum += 1
        debias_weight = 1 - math.pow(weight, num_accum)
        smoothed_val = s_t / debias_weight
        smoothed_ts.append(smoothed_val)

    return pd.Series(smoothed_ts, name=ts_name, index=ts_index)
