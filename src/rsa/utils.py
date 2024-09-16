"""Utility functions related to RSA.

Functions:
    * get_upper_tri_matrix(sq_matrix): Extract the upper triangular part
        of a square matrix.
    * is_vector(x): Check if the input is a ``torch.Tensor`` of
        dimension 1.
    * validate_activations(activations): Validate activations from which
        to compute an RDM.
"""


__all__ = [
    "get_upper_tri_matrix",
    "is_vector",
    "validate_activations"
]

import torch


def get_upper_tri_matrix(sq_matrix: torch.tensor) -> torch.tensor:
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


def validate_activations(activations: torch.Tensor) -> None:
    """Validate activations from which to compute an RDM.

    This function is meant to be called on a matrix of activations from
    which to compute an RDM.  Rows of the matrix are associated with
    different stimuli, while columns correspond to unit activations.
    N stimuli give rise to N * (N - 1) / 2 distinct pairwise distances.
    Hence, there should be activations for at least N >= 3 stimuli.

    Args:
        activations: The matrix of activations from which to compute an
          RDM.  Must be a 2-D tensor of size (N, M), where N >= 3 is the
          number of stimuli (i.e., the batch size), and M >= 2 is the
          number of unit activations per stimulus.

    Raises:
        ValueError: If the ``activations`` tensor does not meet the size
          and dimensionality requirements stated above.
    """

    if not isinstance(activations, torch.Tensor):
        raise ValueError(
            f"'activations' should be of type torch.Tensor, but is of type {type(activations)}.'"
        )

    if activations.dim() != 2:
        raise ValueError(
            f"'activations' should be 2-dimensional, but has {activations.dim()} dimensions."
        )

    if activations.size(dim=0) < 3:
        raise ValueError(
            "'activations' should contain activations for at least 3 stimuli."
        )

    if activations.size(dim=1) < 2:
        raise ValueError(
            "The number of unit activations per stimulus should be at least 2."
        )
