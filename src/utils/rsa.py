"""Utility functions related to RSA.

Functions:
    * get_rsa_loss(compute_name, compute_kwargs, ...): Create an
        instance of the ``rsa_loss`` function w/ def. params.
"""


from functools import partial
from typing import Any, Callable, Dict, Literal, Optional

import torch

from src.rsa.rdm import compute_rdm, compare_rdm
from src.rsa.rsa_loss import rsa_loss
from src.utils.constants import RSA_TRANSFORMS


def get_rsa_loss(
        compute_name: Literal["correlation", "euclidean"],
        compute_kwargs: Dict[str, Any],
        compare_name: Literal["cosine"],
        compare_kwargs: Dict[str, Any],
        weight_rsa_score: float,
        rsa_transform_str: Optional[Literal["abs", "square"]] = None
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create an instance of the ``rsa_loss`` function w/ def. params.

    Args:
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
