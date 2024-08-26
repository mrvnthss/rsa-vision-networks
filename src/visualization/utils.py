"""Utility functions for general visualization purposes.

Functions:
    * save_figure: Save a matplotlib figure.
"""


import warnings
from pathlib import Path

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
