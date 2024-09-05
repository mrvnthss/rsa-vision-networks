"""Utility functions related to color.

Functions:
    * get_color(base_color, tint=0, alpha=1): Obtain a modified version
        (tint & alpha) of a base color.
    * visualize_colors(tint=0, alpha=1): Visualize all (modified) colors
        next to each other.
"""


from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.utils.constants import COLORS


def get_color(
        base_color: str,
        tint: Optional[float] = 0,
        alpha: Optional[float] = 1
) -> np.ndarray:
    """Obtain a modified version (tint & alpha) of a base color.

    Args:
        base_color: The base color from the ``COLORS`` dictionary.
        tint: The relative amount of tint to add.
        alpha: The alpha value of the color for transparency.

    Returns:
        The modified version of the chosen base color.

    Raises:
        ValueError: If the ``base_color`` is not one of the keys of the
          ``COLORS`` dictionary or if either the ``tint`` or the
          ``alpha`` is outside the [0, 1] range.
    """

    if base_color not in COLORS:
        raise ValueError(
            "'base_color' should be one of the keys of the 'COLORS' dictionary, "
            f"but got {base_color}."
        )

    if not 0 <= tint <= 1:
        raise ValueError(
            f"'tint' should be between 0 and 1, but got {tint}."
        )

    if not 0 <= alpha <= 1:
        raise ValueError(
            f"'alpha' should be between 0 and 1, but got {alpha}."
        )

    base_color = COLORS[base_color]
    modified_color = base_color + (1 - base_color) * tint

    if alpha < 1:
        modified_color = np.append(modified_color, alpha)

    return modified_color


def visualize_colors(
        tint: Optional[float] = 0,
        alpha: Optional[float] = 1
) -> None:
    """Visualize all (modified) colors next to each other.

    Args:
        tint: The relative amount of tint to add to each color.
        alpha: The alpha value of each color for transparency.

    Raises:
        ValueError: If either the ``tint`` or the ``alpha`` is outside
          the [0, 1] range.
    """

    if not 0 <= tint <= 1:
        raise ValueError(
            f"'tint' should be between 0 and 1, but got {tint}."
        )

    if not 0 <= alpha <= 1:
        raise ValueError(
            f"'alpha' should be between 0 and 1, but got {alpha}."
        )

    modified_colors = {
        color_name: get_color(color_name, tint=tint, alpha=alpha)
        for color_name in COLORS
    }

    _, ax = plt.subplots(figsize=(5, 3))
    for i, (color_name, color_value) in enumerate(modified_colors.items()):
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=color_value, label=color_name))
        ax.text(i + 0.5, -0.02, color_name, ha="center", va="top", rotation=90, fontsize=11)
    ax.set_xlim(0, len(modified_colors))
    ax.axis("off")
    plt.tight_layout()
    plt.show()
