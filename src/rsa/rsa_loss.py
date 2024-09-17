"""A weighted loss combining cross-entropy and repr. similarity."""


from typing import Callable, Optional, Tuple

import torch
from torch.nn import functional as F

from src.rsa.rdm import validate_activations


def rsa_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        activations1: torch.Tensor,
        activations2: torch.Tensor,
        method_compute: Callable[[torch.Tensor], torch.Tensor],
        method_compare: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        weight_rsa_score: float,
        rsa_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A weighted loss combining cross-entropy and repr. similarity.

    Note:
        Both, the ``method_compute`` and the ``method_compare``
        functions should take as arguments only the activations and
        RDMs, respectively.  Additional arguments must already be
        pre-bound to these functions using the ``partial`` function of
        the ``functools`` module.

    Args:
        predictions: The model predictions.
        targets: The target values.
        activations1: First set of activations.  Must be a 2-D tensor of
          size (N, M), where N >= 3 is the number of stimuli and M >= 2
          is the number of unit activations per stimulus.
        activations2: Second set of activations.  Must be a 2-D tensor
          of size (N, K), where N >= 3 is the number of stimuli and
          K >= 2 is the number of unit activations per stimulus.
        method_compute: The method used to compute the RDMs from the
          activations.
        method_compare: The method used to compare RDMs.
        weight_rsa_score: The weight attributed to the (transformed) RSA
          score in the weighted loss combination.
        rsa_transform: The transformation to apply to the raw RSA score.

    Returns:
        A tuple consisting of the raw RSA score between the two networks
        and the weighted loss.

    Raises:
        ValueError: If ``weight_rsa_score`` is outside the range [0, 1]
          or if one of the activation tensors does not meet the size and
          dimensionality requirements.
    """

    if not 0 <= weight_rsa_score <= 1:
        raise ValueError(
            "'weight_rsa_score' should be a float in the range [0, 1], "
            f"but got {weight_rsa_score}."
        )

    # Check activation tensors
    validate_activations(activations1)
    validate_activations(activations2)
    if activations1.size(dim=0) != activations2.size(dim=0):
        raise ValueError(
            "'activations1' and 'activations2' should have the same number of rows."
        )

    # Compute cross-entropy loss
    loss = F.cross_entropy(predictions, targets)

    # Compute RSA score
    rsa_score = method_compare(
        method_compute(activations1),
        method_compute(activations2)
    )

    # Transform RSA score & compute weighted loss
    rsa_score_transformed = rsa_score if rsa_transform is None else rsa_transform(rsa_score)
    weighted_loss = weight_rsa_score * rsa_score_transformed + (1 - weight_rsa_score) * loss

    return rsa_score, weighted_loss
