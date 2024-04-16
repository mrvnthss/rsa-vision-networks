"""This module provides a wrapper for torchvision's ToDtype class.

The ToDtypeWrapper class wraps torchvision.transforms.v2.ToDtype to
provide compatibility with the Hydra framework.
"""


from typing import Dict, Union

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms


class ToDtypeWrapper(transforms.Transform):
    """A wrapper for the ToDtype class in torchvision.transforms.v2.

    This class accepts a dtype as a string or a dictionary and a scale
    parameter.  If dtype is a string, it is converted to the
    corresponding torch.dtype.  If dtype is a dictionary, its keys
    should be from torchvision.tv_tensors and its values should be
    strings that can be converted to torch.dtype.  The converted
    dtype(s) and scale parameter are then passed to the ToDtype class.

    Parameters:
        dtype: The dtype to convert to. If it's a string, it is
          converted to the corresponding torch.dtype.  If it's a
          dictionary, its keys should be from torchvision.tv_tensors and
          its values should be strings that can be converted to
          torch.dtype.
        scale: Whether to scale the values for images or videos.

    Attributes:
        dtype_map: A mapping from string dtypes to torch dtypes.
        tv_tensor_map: A mapping from torchvision.tv_tensors to their
          corresponding classes.
        to_dtype: An instance of torchvision.transforms.v2.ToDtype.
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

    def __init__(self, dtype: Union[str, Dict[str, str]], scale: bool) -> None:
        super().__init__()
        # Assert that dtype is either a string from dtype_map or
        # a dict with keys from tv_tensor_map and values from dtype_map
        if isinstance(dtype, str):
            assert dtype in self.dtype_map
        elif isinstance(dtype, dict):
            assert all(k in self.tv_tensor_map for k in dtype)
            assert all(v in self.dtype_map for v in dtype.values())
        else:
            raise ValueError(f"dtype must be a str or dict, got {type(dtype)} instead")

        if isinstance(dtype, str):
            dtype = self.dtype_map[dtype]
        else:
            dtype = {self.tv_tensor_map[k]: self.dtype_map[v] for k, v in dtype.items()}

        # Call the class's constructor that is being wrapped
        self.to_dtype = transforms.ToDtype(dtype, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_dtype(x)
