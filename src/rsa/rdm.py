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

from typing import Any, Callable, Dict, Literal, Optional

import torch
from fast_soft_sort.pytorch_ops import soft_rank
from torch.nn import functional as F


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

    # NOTE: The Pearson correlation of two vectors is the same as the cosine similarity between the
    #       centered vectors.
    rdm1 = rdm1 - rdm1.mean()
    rdm2 = rdm2 - rdm2.mean()
    return F.cosine_similarity(rdm1, rdm2, dim=0)


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

    return F.cosine_similarity(rdm1, rdm2, dim=0)


def _compare_rdm_spearman(
        rdm1: torch.Tensor,
        rdm2: torch.Tensor,
        differentiable: bool = False,
        regularization_strength: Optional[float] = None,
        regularization: Optional[Literal["l2", "kl"]] = None
) -> torch.Tensor:
    """Compare two RDMs using Spearman rank correlation.

    Note:
        When computing (true) ranks of the dissimilarities for each RDM,
        ties are assigned the average of the ranks that would have been
        assigned to each value.

    Args:
        rdm1: The first RDM in vectorized form.
        rdm2: The second RDM in vectorized form.
        differentiable: Whether to use soft ranks instead of true ranks.
        regularization_strength: The regularization strength to be used.
          The smaller this number, the closer the values are to the true
          ranks.  Must be specified when ``differentiable`` is True.
        regularization: The regularization method to be used.  Must be
          specified when ``differentiable`` is True.

    Returns:
        The Spearman rank correlation between the two RDMs (possibly
        computed using soft ranks).
    """

    if differentiable:
        if regularization_strength is None:
            raise ValueError(
                "'regularization_strength' must not be None when 'differentiable' is True."
            )
        if regularization is None:
            raise ValueError(
                "'regularization' must not be None when 'differentiable' is True."
            )
        if regularization not in ["l2", "kl"]:
            raise ValueError(
                f"'regularization' should be either 'l2' or 'kl', but got {regularization}."
            )

        # NOTE: The ``soft_rank`` function expects and outputs 2-D tensors, hence the
        #       (un-)squeezing operations.  Also, the ``soft_rank`` function only accepts tensors
        #       on the CPU, a TypeError is raised for tensors on cuda/mps.
        _device = rdm1.device
        rdm1_ranks = soft_rank(
            rdm1.unsqueeze(dim=0).cpu(),
            regularization_strength=regularization_strength,
            regularization=regularization
        ).to(_device)
        rdm2_ranks = soft_rank(
            rdm2.unsqueeze(dim=0).cpu(),
            regularization_strength=regularization_strength,
            regularization=regularization
        ).to(_device)
        rdm1_ranks = rdm1_ranks.squeeze(dim=0)
        rdm2_ranks = rdm2_ranks.squeeze(dim=0)
    else:
        rdm1_ranks = _rank_data(rdm1)
        rdm2_ranks = _rank_data(rdm2)

    return _compare_rdm_correlation(rdm1_ranks, rdm2_ranks)


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
        center_activations: Whether to center the activations across
          units per stimulus before computing distances.
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


def _rank_data(data: torch.Tensor) -> torch.Tensor:
    """Assign ranks to data, with ties resolved by averaging.

    Args:
        data: The data tensor for which to assign ranks.

    Returns:
        The ranks of the data tensor, with ties resolved by averaging.
        Ranks are 1-based and have the same dtype and device as the
        input tensor.
    """

    if not _is_vector(data):
        raise ValueError("'data' should be a 1-D tensor.")

    # Determine ordinal ranks
    n = data.numel()
    rank = torch.empty_like(data)
    idx = data.argsort()
    rank[idx[:n]] = torch.arange(1, n + 1, dtype=data.dtype, device=data.device)

    # Identify duplicate entries in ``data`` tensor
    unique_values, counts = torch.unique(data, return_counts=True)
    duplicate_values = unique_values[counts > 1]

    # Convert ordinal ranks to average ranks for duplicate entries
    for value in duplicate_values:
        mask = data == value
        rank[mask] = rank[mask].float().mean()

    return rank


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
            "Both 'rdm1' and 'rdm2' should be 1-D tensors."
        )

    if not rdm1.numel() == rdm2.numel():
        raise ValueError(
            "The two RDMs should have the same number of elements."
        )
