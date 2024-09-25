"""Functions related to representational dissimilarity matrices.

Functions:
    * compare_rdm(rdm1, rdm2, method, **kwargs): Compare two RDMs using
        the specified method.
    * compute_rdm(activations, method, **kwargs): Compute an RDM using
        the specified method.
    * validate_activations(activations): Validate activations from which
        to compute an RDM.
"""


__all__ = [
    "compare_rdm",
    "compute_rdm",
    "validate_activations"
]

from typing import Any, Callable, Dict, Literal

import torch
from torchmetrics.functional.regression import spearman_corrcoef


def compare_rdm(
    rdm1: torch.Tensor,
    rdm2: torch.Tensor,
    method: str,
    **kwargs: Any
) -> torch.Tensor:
    """Compare two RDMs using the specified method.

    Available methods:
        * "correlation": Pearson correlation.
        * "cosine": Cosine similarity.
        * "spearman": Spearman rank correlation.

    Args:
        rdm1: The first RDM in vectorized form.
        rdm2: The second RDM in vectorized form.
        method: The method to use for comparing the RDMs.
        **kwargs: Additional keyword arguments to pass to the specific
          function used to compare the RDMs.

    Returns:
        The similarity between the two RDMs based on the specified
        method.

    Raises:
        If the ``method`` is not one of the available options.
    """

    methods: dict[str, Callable[..., torch.Tensor]] = {
        "correlation": _compare_rdm_correlation,
        "cosine": _compare_rdm_cosine,
        "spearman": _compare_rdm_spearman
    }

    if method not in methods:
        raise ValueError(
            f"'method' should be one of {list(methods.keys())}, but got {method}."
        )

    _validate_rdms(rdm1, rdm2)
    return methods[method](rdm1, rdm2, **kwargs)


def compute_rdm(
    activations: torch.Tensor,
    method: str,
    **kwargs: Any
) -> torch.Tensor:
    """Compute an RDM using the specified method.

    Available methods:
        * "euclidean": Euclidean distance.
        * "correlation": Pearson correlation distance.

    Args:
        activations: The matrix of activations from which to compute the
        RDM.  Must be a 2-D tensor of size (N, M), where N >= 3 is the
          number of stimuli, and M >= 2 is the number of unit
          activations per stimulus.
        method: The method to use for computing the RDM.
        **kwargs: Additional keyword arguments to pass to the specific
          function used to compute the RDM.

    Returns:
        The RDM (in vectorized form) computed from the data using the
          specified method.

    Raises:
        If the ``method`` is not one of the available options.
    """

    methods: Dict[str, Callable[..., torch.Tensor]] = {
        "euclidean": _compute_rdm_euclidean,
        "correlation": _compute_rdm_correlation,
    }

    if method not in methods:
        raise ValueError(
            f"'method' should be one of {list(methods.keys())}, but got {method}."
        )

    validate_activations(activations)
    return methods[method](activations, **kwargs)


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


def _compare_rdm_correlation(
        rdm1: torch.Tensor,
        rdm2: torch.Tensor
) -> torch.Tensor:
    """Compare two RDMs using Pearson correlation.

    Args:
        rdm1: The first RDM in vectorized form.
        rdm2: The second RDM in vectorized form.

    Returns:
        The Pearson correlation between the two RDMs.
    """

    # NOTE: The Pearson correlation of two vectors x and y is the same as the cosine similarity
    #       between the centered versions of x and y.
    rdm1 = rdm1 - rdm1.mean()
    rdm2 = rdm2 - rdm2.mean()
    return _cosine_similarity(rdm1, rdm2)


def _compare_rdm_cosine(
        rdm1: torch.Tensor,
        rdm2: torch.Tensor
) -> torch.Tensor:
    """Compare two RDMs using cosine similarity.

    Args:
        rdm1: The first RDM in vectorized form.
        rdm2: The second RDM in vectorized form.

    Returns:
        The cosine similarity between the two RDMs.
    """

    return _cosine_similarity(rdm1, rdm2)


def _compare_rdm_spearman(
        rdm1: torch.Tensor,
        rdm2: torch.Tensor
) -> torch.Tensor:
    """Compare two RDMs using Spearman rank correlation.

    Note:
        When computing ranks of the dissimilarities for each RDM, ties
        are assigned the average of the ranks that would have been
        assigned to each value.  This mimics the default behavior of the
        ``scipy.stats.rankdata`` function.

    Args:
        rdm1: The first RDM in vectorized form.
        rdm2: The second RDM in vectorized form.

    Returns:
        The Spearman rank correlation between the two RDMs.
    """

    spearman_correlation = spearman_corrcoef(rdm1, rdm2)
    return spearman_correlation


def _compute_rdm_correlation(
    activations: torch.Tensor
) -> torch.Tensor:
    """Compute an RDM using Pearson correlation distance.

    Note:
        This function returns only the upper triangular matrix of the
        computed RDM in vectorized form (row-major order).

    Args:
        activations: The matrix of activations from which to compute the
          RDM.  Must be a 2-D tensor of size (N, M), where N >= 3 is the
          number of stimuli, and M >= 2 is the number of unit
          activations per stimulus.

    Returns:
        The RDM (in vectorized form) computed from the data using
        Pearson correlation distance.
    """

    return 1 - _get_upper_tri_matrix(torch.corrcoef(activations))


def _compute_rdm_euclidean(
        activations: torch.Tensor,
        center_activations: bool = False,
        normalize_distances: bool = True,
        distance_type: Literal["squared", "non-squared"] = "squared"
) -> torch.Tensor:
    """Compute an RDM using Euclidean distance.

    Note:
        This function returns only the upper triangular matrix of the
        computed RDM in vectorized form (row-major order).

    Args:
        activations: The matrix of activations from which to compute the
          RDM.  Must be a 2-D tensor of size (N, M), where N >= 3 is the
          number of stimuli, and M >= 2 is the number of unit
          activations per stimulus.
        center_activations: Whether to center the activations for each
          stimulus before computing distances.
        normalize_distances: Whether to normalize the squared pairwise
          distances by the number M of unit activations per stimulus.
        distance_type: Whether to return squared or non-squared
          distances.

    Returns:
        The RDM (in vectorized form) computed from the data using
        Euclidean distance.

    Raises:
        If the ``distance_type`` is neither "squared" nor "non-squared".
    """

    if distance_type not in ["squared", "non-squared"]:
        raise ValueError(
            "'distance_type' should be either 'squared' or 'non-squared', "
            f"but got {distance_type}."
        )

    if center_activations:
        activations = activations - activations.mean(dim=1, keepdim=True)

    # Compute squared pairwise distances
    norms_squared = torch.sum(torch.square(activations), dim=1, keepdim=True)
    distances_squared = norms_squared + norms_squared.T - 2 * torch.mm(activations, activations.T)

    # Clamp negative values (potential numerical inaccuracies) and extract RDM
    rdm = _get_upper_tri_matrix(torch.clamp(distances_squared, min=0.0))

    if normalize_distances:
        rdm = rdm / activations.size(dim=1)

    if distance_type == "non-squared":
        rdm = torch.sqrt(rdm)

    return rdm


def _cosine_similarity(
        vec1: torch.Tensor,
        vec2: torch.Tensor
) -> torch.Tensor:
    """Compute the cosine similarity between two vectors.

    Args:
        vec1: The first vector.
        vec2: The second vector.

    Returns:
        The cosine similarity between the two vectors.

    Raises:
        ValueError: If one of the inputs is not a 1-D tensor or if the
          two vectors do not have the same number of elements.
    """

    if not (_is_vector(vec1) and _is_vector(vec2)):
        raise ValueError(
            "Both 'vec1' and 'vec2' should be tensors of dimension 1."
        )

    if not vec1.numel() == vec2.numel():
        raise ValueError(
            "The two vectors should have the same number of elements."
        )

    norm1 = torch.linalg.vector_norm(vec1, ord=2)
    norm2 = torch.linalg.vector_norm(vec2, ord=2)

    if norm1 == 0 or norm2 == 0:
        return torch.tensor([0.0])

    return torch.dot(vec1, vec2) / (norm1 * norm2)


def _get_upper_tri_matrix(sq_matrix: torch.Tensor) -> torch.Tensor:
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


def _is_vector(x: torch.Tensor) -> bool:
    """Check if the input is a ``torch.Tensor`` of dimension 1."""

    return isinstance(x, torch.Tensor) and x.dim() == 1


def _validate_rdms(
        rdm1: torch.Tensor,
        rdm2: torch.Tensor
) -> None:
    """Validate RDMs from which to compute a similarity score.

    Args:
        rdm1: The first RDM in vectorized form.
        rdm2: The second RDM in vectorized form.

    Raises:
        ValueError: If one of the RDMs is not in vectorized form or if
          the two RDMs do not have the same number of elements.
    """

    if not (_is_vector(rdm1) and _is_vector(rdm2)):
        raise ValueError(
            "Both 'rdm1' and 'rdm2' should be tensors of dimension 1."
        )

    if not rdm1.numel() == rdm2.numel():
        raise ValueError(
            "The two RDMs should have the same number of elements."
        )
