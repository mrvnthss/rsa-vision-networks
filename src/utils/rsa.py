"""Utility functions related to RSA.

Functions:
    * get_rsa_loss(label_smoothing, compute_name, ...): Create an
        instance of the ``rsa_loss`` function w/ def. params.
"""


from functools import partial
from typing import Callable

import torch

from src.config import ReprSimilarityConf
from src.rsa.rdm import compute_rdm, compare_rdm
from src.rsa.rsa_loss import rsa_loss
from src.utils.constants import RSA_TRANSFORMS


def get_rsa_loss(
        repr_similarity_params: ReprSimilarityConf
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create an instance of the ``rsa_loss`` function w/ def. params.

    Args:
        repr_similarity_params: The parameters specifying the RSA
          pipeline (computing & comparing RDMs, etc.).

    Returns:
        The instance of the ``rsa_loss`` function with parameters as
        chosen.
    """

    is_transform_specified = (
            "rsa_transform" in repr_similarity_params
            and repr_similarity_params.rsa_transform is not None
    )
    if is_transform_specified:
        if repr_similarity_params.rsa_transform not in RSA_TRANSFORMS:
            raise ValueError(
                f"'rsa_transform' should be one of {list(RSA_TRANSFORMS.keys())}, "
                f"but got {repr_similarity_params.rsa_transform}."
            )
        rsa_transform = RSA_TRANSFORMS[repr_similarity_params.rsa_transform]
    else:
        rsa_transform = None

    rsa_loss_fn = partial(
        rsa_loss,
        method_compute=partial(
            compute_rdm,
            method=repr_similarity_params.compute_rdm.name,
            **repr_similarity_params.compute_rdm.kwargs
        ),
        method_compare=partial(
            compare_rdm,
            method=repr_similarity_params.compare_rdm.name,
            **repr_similarity_params.compare_rdm.kwargs
        ),
        weight_rsa_score=repr_similarity_params.weight_rsa_score,
        rsa_transform=rsa_transform
    )

    return rsa_loss_fn
