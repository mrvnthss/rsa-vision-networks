"""Utility functions for visualizing datasets.

Functions:
    * create_sprite: Combine individual images into a sprite.
    * get_samples: Grab a subset of samples from each class in the
        dataset.
"""


from pathlib import Path

import numpy as np
from PIL import Image


def create_sprite(
        images: np.ndarray,
        n_rows: int,
        n_cols: int
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

    Returns:
        The sprite made up of the individual images.
    """

    if images.ndim == 3:
        num_images, height, width, channels = (*images.shape, 1)
    else:
        num_images, height, width, channels = images.shape

    sprite = np.squeeze(
        np.zeros((height * n_rows, width * n_cols, channels), dtype=np.uint8)
    )

    for row in range(n_rows):
        for col in range(n_cols):
            next_img_idx = row * n_cols + col
            if next_img_idx < num_images:
                next_img = images[next_img_idx]
                sprite[row * height:(row + 1) * height, col * width:(col + 1) * width] = next_img

    return sprite


def get_samples(
        data_dir: str,
        num_samples: int,
        interleave_classes: bool = False,
        train: bool = True,
        random_seed: int = 42
) -> np.ndarray:
    """Grab a subset of samples from each class in the dataset.

    Args:
        data_dir: The path to the directory containing the processed
          dataset.
        num_samples: The number of samples to select per class.
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

    # Use seeded rng for reproducibility
    rng = np.random.default_rng(random_seed)

    # Collect samples from each class
    data_dir = Path(data_dir) / ("train" if train else "test")
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    samples = {}
    for class_dir in class_dirs:
        class_name = class_dir.name
        img_paths = list(class_dir.iterdir())
        selected_indices = rng.choice(len(img_paths), num_samples, replace=False)
        samples[class_name] = np.array([
            np.array(Image.open(img_paths[idx])) for idx in selected_indices
        ])

    # Determine dimension and shape of output array
    img = samples[next(iter(samples))][0]
    if img.ndim == 2:
        height, width, channels = (*img.shape, 1)
    else:
        height, width, channels = img.shape

    samples_np = np.squeeze(
        np.empty((num_samples * len(samples), height, width, channels), dtype=np.uint8)
    )

    # Combine images from dictionary of lists into NumPy array
    if interleave_classes:
        for img_idx in range(num_samples):
            for class_idx, class_name in enumerate(samples):
                samples_np[img_idx * len(samples) + class_idx] = samples[class_name][img_idx]
    else:
        for class_idx, class_name in enumerate(samples):
            for img_idx in range(num_samples):
                samples_np[class_idx * num_samples + img_idx] = samples[class_name][img_idx]

    return samples_np
