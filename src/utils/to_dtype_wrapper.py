from typing import Dict, Union

import torch
import torchvision


class ToDtypeWrapper(torchvision.transforms.v2.Transform):
    """
    A wrapper class for torchvision.transforms.v2.ToDtype.

    This class accepts a dtype as a string or a dictionary and a scale
    parameter.  If dtype is a string, it is converted to the
    corresponding torch.dtype.  If dtype is a dictionary, its keys
    should be from torchvision.tv_tensors and its values should be
    strings that can be converted to torch.dtype.  The converted
    dtype(s) and scale parameter are then passed to the ToDtype class.

    Args:
        dtype (Union[str, Dict[str, str]]): The dtype to convert to.
            If it's a string, it is converted to the corresponding
            torch.dtype.  If it's a dictionary, its keys should be from
            torchvision.tv_tensors and its values should be strings that
            can be converted to torch.dtype.
        scale (bool): Whether to scale the values for images or videos.

    Attributes:
        to_dtype (ToDtype): An instance of
            torchvision.transforms.v2.ToDtype.
    """
    def __init__(self, dtype: Union[str, Dict[str, str]], scale: bool) -> None:
        super().__init__()

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
            "Image": torchvision.tv_tensors.Image,
            "Video": torchvision.tv_tensors.Video,
            "BoundingBoxes": torchvision.tv_tensors.BoundingBoxes,
            "Mask": torchvision.tv_tensors.Mask
        }

        # Assert that dtype is either a string from dtype_map.keys() or
        # a dict with keys from tv_tensor_map.keys() and values from dtype_map.keys()
        if isinstance(dtype, str):
            assert dtype in dtype_map.keys()
        elif isinstance(dtype, dict):
            assert all([k in tv_tensor_map.keys() for k in dtype.keys()])
            assert all([v in dtype_map.keys() for v in dtype.values()])
        else:
            raise ValueError(f"dtype must be a str or dict, got {type(dtype)} instead")

        if isinstance(dtype, str):
            dtype = dtype_map[dtype]
        else:  # dict
            dtype = {tv_tensor_map[k]: dtype_map[v] for k, v in dtype.items()}

        # Call the class's constructor that we are wrapping
        self.to_dtype = torchvision.transforms.v2.ToDtype(dtype, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_dtype(x)
