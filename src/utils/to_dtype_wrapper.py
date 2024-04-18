"""A wrapper for torchvision's ToDtype class.

The ToDtypeWrapper class wraps ``torchvision.transforms.v2.ToDtype`` to
provide compatibility with the Hydra framework.
"""


from typing import Dict, Union

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms


class ToDtypeWrapper(transforms.Transform):
    """A wrapper for the ``ToDtype`` class in torchvision.transforms.v2.

    Accepts a string or a dict specifying the PyTorch data type to
    convert to and a boolean indicating whether to scale the values for
    images or videos.

    Params:
        dtype: The dtype to convert to.  If a string is passed (e.g.,
          "float32"), only images and videos will be converted to the
          corresponding dtype (e.g., ``torch.float32``).  A dict can be
          passed to specify per-tv_tensor conversions.
        scale: Whether to scale the values for images or videos.
          Default: False.

    Attributes:
        dtype_map: A mapping from strings to ``torch.dtype``.
        tv_tensor_map: A mapping from strings to ``tv_tensors``.
        to_dtype: An instance of ``torchvision.transforms.v2.ToDtype``.
    """

    dtype_map = {
        "float32": torch.float32,
        "float": torch.float,
        "float64": torch.float64,
        "double": torch.double,
        "complex64": torch.complex64,
        "cfloat": torch.cfloat,
        "complex128": torch.complex128,
        "cdouble": torch.cdouble,
        "float16": torch.float16,
        "half": torch.half,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "short": torch.short,
        "int32": torch.int32,
        "int": torch.int,
        "int64": torch.int64,
        "long": torch.long,
        "bool": torch.bool
    }

    tv_tensor_map = {
        "Image": tv_tensors.Image,
        "Video": tv_tensors.Video,
        "BoundingBoxes": tv_tensors.BoundingBoxes,
        "Mask": tv_tensors.Mask
    }

    def __init__(
            self,
            dtype: Union[str, Dict[str, str]],
            scale: bool = False
    ) -> None:
        super().__init__()
        if isinstance(dtype, str):
            assert dtype in self.dtype_map
        elif isinstance(dtype, dict):
            assert all(k in self.tv_tensor_map and v in self.dtype_map for k, v in dtype.items())
        else:
            raise ValueError(f"dtype must be a str or dict, got {type(dtype)} instead")

        # Replace strings with actual torch.dtype and torchvision.tv_tensors instances
        if isinstance(dtype, str):
            dtype = self.dtype_map[dtype]
        else:
            dtype = {self.tv_tensor_map[k]: self.dtype_map[v] for k, v in dtype.items()}

        self.to_dtype = transforms.ToDtype(dtype, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_dtype(x)
