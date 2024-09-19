"""Utility functions to save visualizations.

Functions:
    * save_figure(fig, f_path, dpi=300): Save a matplotlib figure.
    * save_image(img, f_path, quality=100): Save a PIL image.
"""


import warnings
from pathlib import Path
from typing import Union

from PIL import Image
from matplotlib.figure import Figure


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


def save_image(
        img: Image.Image,
        f_path: Union[str, Path],
        quality: int = 100
) -> None:
    """Save a PIL image.

    Args:
        img: The PIL image object to be saved.
        f_path: The file path where the image should be saved.
        quality: The quality of the saved image (0-100).
    """

    f_path = Path(f_path)
    f_path.parent.mkdir(parents=True, exist_ok=True)

    if f_path.exists():
        warnings.warn(f"File {f_path} already exists! Image not saved.")
    else:
        img.save(f_path, quality=quality)
        print(f"Image saved successfully as {f_path}.")
