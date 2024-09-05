"""Utility functions related to color.

Functions:
    * get_color(base_color, tint=0, alpha=1): Obtain a modified version
        (tint & alpha) of a base color.
"""


from typing import Optional

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
