"""Utility functions related to RSA.

Functions:
    * get_rsa_loss(label_smoothing, compute_name, ...): Create an
        instance of the ``rsa_loss`` function w/ def. params.
"""


from functools import partial
from typing import Any, Callable, Dict, Literal, Optional

import torch

from src.config import CompareRDMName, ComputeRDMName, RSATransformName
from src.rsa.rdm import compute_rdm, compare_rdm
from src.rsa.rsa_loss import rsa_loss
from src.utils.constants import RSA_TRANSFORMS


def get_rsa_loss(
        label_smoothing: float,
        compute_name: ComputeRDMName,
        compute_kwargs: Dict[str, Any],
        compare_name: CompareRDMName,
        compare_kwargs: Dict[str, Any],
        weight_rsa_score: float,
        rsa_transform_str: Optional[RSATransformName] = None
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create an instance of the ``rsa_loss`` function w/ def. params.

    Args:
        label_smoothing: The amount of smoothing when computing the
          cross-entropy loss.
        compute_name: The method to use for computing the RDMs.
        compute_kwargs: Additional keyword arguments to pass to the
          specific function used to compute the RDMs.
        compare_name: The method to use for comparing the RDMs.
        compare_kwargs: Additional keyword arguments to pass to the
          specific function used to compare the RDMs.
        weight_rsa_score: The weight attributed to the (transformed) RSA
          score in the weighted loss combination.
        rsa_transform_str: The name of the transformation to apply to
          the raw RSA score, if any.

    Returns:
        The instance of the ``rsa_loss`` function with parameters as
        chosen.
    """

    if rsa_transform_str is not None:
        if rsa_transform_str not in RSA_TRANSFORMS:
            raise ValueError(
                f"'rsa_transform_str' should be one of {list(RSA_TRANSFORMS.keys())}, "
                f"but got {rsa_transform_str}."
            )
        rsa_transform = RSA_TRANSFORMS[rsa_transform_str]
    else:
        rsa_transform = None

    rsa_loss_fn = partial(
        rsa_loss,
        label_smoothing=label_smoothing,
        method_compute=partial(
            compute_rdm,
            method=compute_name,
            **compute_kwargs
        ),
        method_compare=partial(
            compare_rdm,
            method=compare_name,
            **compare_kwargs
        ),
        weight_rsa_score=weight_rsa_score,
        rsa_transform=rsa_transform
    )

    return rsa_loss_fn
