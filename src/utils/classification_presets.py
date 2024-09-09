"""A typical transform for image classification tasks.

Adapted from
https://github.com/pytorch/vision/tree/main/references/classification.
"""


from typing import Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


class ClassificationPresets:
    """Provides a typical transform for image classification tasks.

    Attributes:
        transform: The transform that's applied to the input image when
          the instance is called.
    """

    def __init__(
            self,
            mean: Sequence[float],
            std: Sequence[float],
            crop_size: Union[int, Sequence[int]],
            crop_scale: Tuple[float, float] = (0.08, 1.0),
            crop_ratio: Tuple[float, float] = (3/4, 4/3),
            flip_prob: float = 0.5,
            resize_size: Optional[Union[int, Sequence[int]]] = None,
            is_training: bool = True,
            interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR
    ) -> None:
        """Initialize the transform.

        Args:
            mean: The mean values for normalization.
            std: The standard deviation values for normalization.
            crop_size: The desired output size of the random/center
              crop.  This should match the expected input size of the
              model.
            crop_scale: The lower and upper bounds for the area of the
              random crop, before resizing.  The scale is defined with
              respect to the area of the original image.  Has no effect
              when ``is_training`` is False.
            crop_ratio: The lower and upper bounds for the aspect ratio
              of the random crop, before resizing.  Has no effect when
              ``is_training`` is False.
            flip_prob: The probability of flipping images horizontally.
              Has no effect when ``is_training`` is False.
            resize_size: The desired output size when resizing prior to
              performing a center crop.  Only used when ``is_training``
              is False.
            is_training: Whether to initialize the training or
              validation transforms.
            interpolation: The interpolation mode to use when resizing
              images.
        """

        transforms = []

        if is_training:
            transforms.append(
                T.RandomResizedCrop(
                    size=crop_size,
                    scale=crop_scale,
                    ratio=crop_ratio,
                    interpolation=interpolation,
                    antialias=True
                ))
            if flip_prob > 0:
                transforms.append(T.RandomHorizontalFlip(flip_prob))
        else:
            transforms += [
                T.Resize(
                    size=resize_size,
                    interpolation=interpolation,
                    antialias=True
                ),
                T.CenterCrop(crop_size)
            ]

        transforms += [
            T.PILToTensor(),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
            T.ToPureTensor()
        ]

        self.transform = T.Compose(transforms)

    def __call__(
            self,
            img: Image.Image
    ) -> torch.Tensor:
        return self.transform(img)
