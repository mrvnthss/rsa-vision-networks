"""Utility functions for visualizing datasets.

Functions:
    * create_sprite(images, n_rows, n_cols, by_row=True): Combine
        individual images into a sprite.
    * load_class_samples_as_array(data_dir, ...): Load a fixed number of
        samples from each class in the dataset.
    * load_class_samples_as_dict(data_dir, ...): Load a fixed number of
        samples from each class in the dataset.
    * visualize_crop(img, crop_scale, ...): Visualize a random crop of
        an image.
"""


__all__ = [
    "create_sprite",
    "load_class_samples_as_array",
    "load_class_samples_as_dict",
    "visualize_crop"
]

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.v2 import Pad, RandomResizedCrop


def create_sprite(
        images: np.ndarray,
        n_rows: int,
        n_cols: int,
        by_row: bool = True
) -> np.ndarray:
    """Combine individual images into a sprite.

    Adapted from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/helper.py.

    Note:
        This function assumes that the individual images are all of
        equal dimension.  The ``images`` argument is assumed to have
        shape 'samples x height x width x channels' (color images) or
        'samples x height x width' (grayscale images).  No error or
        warning is raised if more images are passed than can be used
        based on the ``n_rows`` and ``n_cols`` arguments.

    Args:
        images: The individual images to combine into a sprite.
        n_rows: The number of rows in the sprite.
        n_cols: The number of columns in the sprite.
        by_row: Whether to create the sprite by row or by column.

    Returns:
        The sprite made up of the individual images.
    """

    num_images, height, width = images.shape[:3]
    channels = images.shape[3] if images.ndim == 4 else 1

    sprite = np.squeeze(
        np.zeros((height * n_rows, width * n_cols, channels), dtype=np.uint8)
    )

    for idx in range(n_rows * n_cols):
        if idx >= num_images:
            break
        row, col = (idx // n_cols, idx % n_cols) if by_row else (idx % n_rows, idx // n_rows)
        next_img = images[idx]
        sprite[row * height:(row + 1) * height, col * width:(col + 1) * width] = next_img

    return sprite


def load_class_samples_as_array(
        data_dir: str,
        num_samples_per_class: int,
        interleave_classes: bool = False,
        train: bool = True,
        random_seed: int = 42
) -> np.ndarray:
    """Load a fixed number of samples from each class in the dataset.

    Note:
        This function assumes that all images have the same shape.

    Args:
        data_dir: The path to the directory containing the processed
          dataset.
        num_samples_per_class: The number of samples to load per class.
        interleave_classes: Whether to return images in blocks by class
          (False) or interleaved (True).
        train: Whether to load images from the training split (True) or
          the test split (False).
        random_seed: The random seed to ensure reproducibility.

    Returns:
        A NumPy array of shape 'samples x height x width x channels'
        (color images) or 'samples x height x width' (grayscale images)
        consisting of the individual images.
    """

    # Load samples from each class
    samples = load_class_samples_as_dict(
        data_dir=data_dir,
        num_samples_per_class=num_samples_per_class,
        train=train,
        random_seed=random_seed
    )

    # Determine dimension and shape of final output array
    img = samples[next(iter(samples))][0]
    height, width = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1

    # Create empty NumPy array to collect all samples
    samples_np = np.squeeze(
        np.empty((num_samples_per_class * len(samples), height, width, channels), dtype=np.uint8)
    )

    # Combine images from dictionary of lists into NumPy array
    if interleave_classes:
        for img_idx in range(num_samples_per_class):
            for class_idx, class_name in enumerate(samples):
                samples_np[
                    img_idx * len(samples) + class_idx
                ] = samples[class_name][img_idx]
    else:
        for class_idx, class_name in enumerate(samples):
            for img_idx in range(num_samples_per_class):
                samples_np[
                    class_idx * num_samples_per_class + img_idx
                ] = samples[class_name][img_idx]

    return samples_np


def load_class_samples_as_dict(
        data_dir: str,
        num_samples_per_class: int,
        train: bool = True,
        random_seed: int = 42
) -> Dict[str, List[np.ndarray]]:
    """Load a fixed number of samples from each class in the dataset.

    Args:
        data_dir: The path to the directory containing the processed
          dataset.
        num_samples_per_class: The number of samples to load per class.
        train: Whether to load images from the training split (True) or
          the test split (False).
        random_seed: The random seed to ensure reproducibility.

    Returns:
        A dictionary mapping names of classes to lists of images from
        that class.
    """

    # Use seeded rng for reproducibility
    rng = np.random.default_rng(random_seed)

    # Load samples from each class
    data_dir = Path(data_dir) / ("train" if train else "test")
    samples = {}
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = str(class_dir.name)
        img_paths = list(class_dir.iterdir())
        selected_indices = rng.choice(len(img_paths), num_samples_per_class, replace=False)
        samples[class_name] = [
            np.array(Image.open(img_paths[idx])) for idx in selected_indices
        ]

    return samples


def visualize_crop(
        img: Image.Image,
        crop_scale: float,
        crop_ratio: float,
        frame_width: int = 1,
        frame_color: Tuple[int] = (255, 0, 0),
        seed: int = 42
) -> Tuple[Image.Image, Image.Image]:
    """Visualize a random crop of an image.

    Args:
        img: The image to crop.
        crop_scale: The relative size of the crop area with respect to
          the area of the original image, before resizing.
        crop_ratio: The aspect ratio of the crop, before resizing.
        frame_width: The width of the frame that's used to highlight the
          cropped area in the original image.
        frame_color: The color of the frame that's used to highlight the
          cropped area in the original image.
        seed: The random seed to ensure reproducibility of the random
          crop.

    Returns:
        A tuple containing the original image with a frame highlighting
        the cropped area and the cropped image (resized to the same size
        as the original image).
    """

    if img.mode != "RGB":
        img = img.convert("RGB")

    # Create transform
    random_resized_crop = RandomResizedCrop(
        size=img.size,
        scale=(crop_scale, crop_scale),
        ratio=(crop_ratio, crop_ratio)
    )

    # Get crop parameters (for visualization purposes)
    torch.manual_seed(seed)
    row, col, height, width = random_resized_crop.get_params(
        img,
        scale=(crop_scale, crop_scale),
        ratio=(crop_ratio, crop_ratio)
    )

    # Highlight region to be cropped in original image
    # NOTE: We draw a bounding box around (!) the region that is to be cropped.  To do so, we pad
    #       the image to prevent the bounding box from being cut off in case the random crop starts
    #       or ends at an edge of the image.
    cropped_img = img.copy()
    img = Pad(padding=frame_width)(img)
    crop_box = [
        col,
        row,
        col + height + 2 * frame_width - 1,
        row + width + 2 * frame_width - 1
    ]
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        crop_box,
        outline=frame_color,
        width=frame_width
    )

    # Crop image
    torch.manual_seed(seed)
    cropped_img = random_resized_crop(cropped_img)
    cropped_img = Pad(padding=frame_width, fill=frame_color)(cropped_img)

    return img, cropped_img
