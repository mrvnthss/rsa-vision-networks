"""Functions related to representational dissimilarity matrices.

Functions:
    * compare_rdm(rdm1, rdm2, method, **kwargs): Compare two RDMs using
        the specified method.
    * compute_rdm(activations, method, **kwargs): Compute an RDM using
        the specified method.
"""


__all__ = [
    "compare_rdm",
    "compute_rdm"
]

from typing import Any, Callable, Dict, Literal

import torch
from torch import linalg as LA

from src.rsa.utils import get_upper_tri_matrix, is_vector, validate_activations


def compare_rdm(
    rdm1: torch.Tensor,
    rdm2: torch.Tensor,
    method: str,
    **kwargs: Any
) -> torch.Tensor:
    """Compare two RDMs using the specified method.

    Available methods:
        * "cosine": Cosine similarity.

    Args:
        rdm1: The first RDM in vectorized form.
        rdm2: The second RDM in vectorized form.
        method: The method to use for comparing the RDMs.
        **kwargs: Additional keyword arguments to pass to the specific
          function used to compare the RDMs.

    Returns:
        The similarity between the two RDMs based on the specified
        method.
    """

    methods: dict[str, Callable[..., torch.Tensor]] = {
        "cosine": _compare_rdm_cosine
    }

    if method not in methods:
        raise ValueError(
            f"'method' should be one of {list(methods.keys())}, but got {method}."
        )

    return methods[method](rdm1, rdm2, **kwargs)


def compute_rdm(
    activations: torch.Tensor,
    method: str,
    **kwargs: Any
) -> torch.Tensor:
    """Compute an RDM using the specified method.

    Available methods:
        * "euclidean": Euclidean distance.
        * "correlation": (Pearson) correlation distance.

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
    """

    methods: Dict[str, Callable[..., torch.Tensor]] = {
        "euclidean": _compute_rdm_euclidean,
        "correlation": _compute_rdm_correlation,
    }

    if method not in methods:
        raise ValueError(
            f"'method' should be one of {list(methods.keys())}, but got {method}."
        )

    return methods[method](activations, **kwargs)


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

    if not (is_vector(rdm1) and is_vector(rdm2)):
        raise ValueError(
            "Both 'rdm1' and 'rdm2' should be tensors of dimension 1."
        )

    cosine_similarity = torch.dot(rdm1, rdm2) / (
            LA.vector_norm(rdm1, ord=2) * LA.vector_norm(rdm2, ord=2)
    )
    return cosine_similarity


def _compute_rdm_correlation(
    activations: torch.Tensor,
    center_activations: bool = True
) -> torch.Tensor:
    """Compute an RDM using (Pearson) correlation distance.

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

    Returns:
        The RDM (in vectorized form) computed from the data using
        (Pearson) correlation distance.
    """

    validate_activations(activations)

    if center_activations:
        activations -= activations.mean(dim=1, keepdim=True)

    rdm = 1 - get_upper_tri_matrix(torch.corrcoef(activations))

    return rdm


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
    """

    validate_activations(activations)

    if distance_type not in ["squared", "non-squared"]:
        raise ValueError(
            "'distance_type' should be either 'squared' or 'non-squared', "
            f"but got {distance_type}."
        )

    if center_activations:
        activations -= activations.mean(dim=1, keepdim=True)

    # Compute squared pairwise distances
    norms_squared = torch.sum(torch.square(activations), dim=1, keepdim=True)
    distances_squared = norms_squared + norms_squared.T - 2 * torch.mm(activations, activations.T)

    # Clamp negative values (potential numerical inaccuracies) and extract RDM
    rdm = get_upper_tri_matrix(torch.clamp(distances_squared, min=0.0))

    if normalize_distances:
        rdm = rdm / activations.size(dim=1)

    if distance_type == "non-squared":
        rdm = torch.sqrt(rdm)

    return rdm
