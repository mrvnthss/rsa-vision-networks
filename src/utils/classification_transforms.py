"""Typical transforms for image classification tasks.

Adapted from:
    https://github.com/pytorch/vision/tree/main/references/classification.

Classes:
    * ClassificationTransformsTrain(mean, std, ...): A composed
        transform for image classification tasks (training).
    * ClassificationTransformsVal(mean, std, ...): A composed transform
        for image classification tasks (validation).
"""


__all__ = [
    "ClassificationTransformsTrain",
    "ClassificationTransformsVal"
]

from typing import Sequence, Union

import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from src.config import CropRatioConf, CropScaleConf


class ClassificationTransformsTrain:
    """A composed transform for image classification tasks (training).

    Attributes:
        transform: The transform that's applied to the input image when
          the instance is called.
    """

    def __init__(
            self,
            mean: Sequence[float],
            std: Sequence[float],
            crop_size: Union[int, Sequence[int]],
            crop_scale: CropScaleConf,
            crop_ratio: CropRatioConf,
            flip_prob: float = 0.5,
            interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR
    ) -> None:
        """Initialize the transform.

        Args:
            mean: The mean values for normalization.
            std: The standard deviation values for normalization.
            crop_size: The desired output size of the random resized
              crop.  This should match the expected input size of the
              model.
            crop_scale: The lower and upper bounds for the area of the
              random crop, before resizing.  The scale is defined with
              respect to the area of the original image.
            crop_ratio: The lower and upper bounds for the aspect ratio
              of the random resized crop, before resizing.
            interpolation: The interpolation mode to use when resizing
              images.
            flip_prob: The probability of flipping images horizontally.
        """

        transforms = [
            T.RandomResizedCrop(
                size=crop_size,
                scale=(crop_scale.lower, crop_scale.upper),
                ratio=(crop_ratio.lower, crop_ratio.upper),
                interpolation=interpolation,
                antialias=True
            )
        ]
        if flip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(flip_prob))

        transforms += [
            T.PILToTensor(),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
            T.ToPureTensor()
        ]

        self.transform = T.Compose(transforms)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class ClassificationTransformsVal:
    """A composed transform for image classification tasks (validation).

    Attributes:
        transform: The transform that's applied to the input image when
          the instance is called.
    """

    def __init__(
            self,
            mean: Sequence[float],
            std: Sequence[float],
            resize_size: Union[int, Sequence[int]],
            crop_size: Union[int, Sequence[int]],
            interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR
    ) -> None:
        """Initialize the transform for validation.

        Args:
            mean: The mean values for normalization.
            std: The standard deviation values for normalization.
            resize_size: The desired output size when resizing prior to
              performing a center crop.
            crop_size: The desired output size of the center crop.  This
              should match the expected input size of the model.
            interpolation: The interpolation mode to use when resizing
              images.
        """

        transforms = [
            T.Resize(
                size=resize_size,
                interpolation=interpolation,
                antialias=True
            ),
            T.CenterCrop(crop_size),
            T.PILToTensor(),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
            T.ToPureTensor()
        ]

        self.transform = T.Compose(transforms)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)
