"""Utility functions related to color.

Functions:
    * get_tinted_color: Get a tinted version of a base color.
"""


from typing import Optional

import numpy as np

from src.utils.constants import COLORS


def get_tinted_color(
        base_color: str,
        tint: Optional[float] = 0
) -> np.ndarray:
    """Get a tinted version of a base color.

    Args:
        base_color: The base color from the ``COLORS`` dictionary.
        tint: The relative amount of tint to add.

    Returns:
        The tinted version of the chosen base color.

    Raises:
        ValueError: If the ``base_color`` is not one of the keys of the
          ``COLORS`` dictionary or if the ``tint`` is outside the [0, 1]
          range.
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

    base_color = COLORS[base_color]
    tinted_color = base_color + (1 - base_color) * tint
    return tinted_color
