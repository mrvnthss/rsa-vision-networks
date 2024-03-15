import torch
from torchvision.transforms import v2


class ToDtypeWrapper:
    """
    A wrapper class for torchvision.transforms.v2.ToDtype.

    This class accepts a dtype as a string and a scale parameter. It
    converts the dtype to the corresponding torch.dtype and passes them
    to the ToDtype class.

    Args:
        dtype (str): The dtype to convert to. The string is converted to
            the corresponding torch.dtype.
        scale (bool): Whether to scale the values for images or videos.
    """
    def __init__(self, dtype: str, scale: bool):
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

        self.inner = v2.ToDtype(dtype=dtype_map[dtype], scale=scale)

    def __call__(self, img):
        return self.inner(img)
