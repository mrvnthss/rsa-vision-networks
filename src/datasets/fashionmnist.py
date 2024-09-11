"""The FashionMNIST dataset by Zalando Research (Xiao et al., 2017)."""


import logging
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader, ImageFolder
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from tqdm import tqdm


class FashionMNIST(ImageFolder):
    """FashionMNIST dataset by Zalando Research (Xiao et al., 2017).

    The class can be used to download, parse, and load the FashionMNIST
    dataset.  The FashionMNIST dataset consists of 70,000 28x28
    grayscale images from 10 classes, with 7,000 images per class.  Per
    class, there are 6,000 training samples and 1,000 test samples.

    Attributes:
        class_to_idx: A dictionary mapping class names to class indices.
        classes: The class labels of the dataset, sorted alphabetically.
        data: A NumPy array containing all images in the dataset if
          ``load_into_memory`` is set to True when initializing the
          dataset.
        data_dir: The path of the "data/" directory containing all
          datasets.
        data_by_split: The names of the raw data files for each dataset.
        imgs: A list of (image path, class index) tuples.
        loader: A function to load a sample given its index.
        logger: A logger instance to record logs.
        mirror: The URL mirror to download the dataset from.
        raw_data: The names and MD5 hashes of the raw data files.
        resources: The names and MD5 hashes of the dataset archives.
        split: The dataset split to load, either "train" or "test".
        split_dir: The directory containing the dataset split.
        target_transform: A function/transform that takes in the target
          and transforms it.
        targets: A list containing the class index for each image in the
          dataset.
        transform: A function/transform that takes in a PIL image and
          returns a transformed version.
    """

    mirror = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310")
    ]

    raw_data = [
        ("train-images-idx3-ubyte", "f4a8712d7a061bf5bd6d2ca38dc4d50a"),
        ("train-labels-idx1-ubyte", "9018921c3c673c538a1fc5bad174d6f9"),
        ("t10k-images-idx3-ubyte", "8181f5470baa50b63fa0f6fddb340f0a"),
        ("t10k-labels-idx1-ubyte", "15d484375f8d13e6eb1aabb0c3f46965")
    ]

    data_by_split = {
        "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
        "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    }

    classes = [
        "Bag",
        "Boot",  # Ankle boot
        "Coat",
        "Dress",
        "Pullover",
        "Sandal",
        "Shirt",
        "Sneaker",
        "T-Shirt",  # T-shirt/top
        "Trouser",
    ]

    def __init__(
            self,
            data_dir: str,
            train: bool = True,
            load_into_memory: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        """Initialize the FashionMNIST dataset.

        Args:
            data_dir: The path of the "data/" directory containing all
              datasets.
            train: Whether to load the training split (True) or the test
              split (False).
            load_into_memory: Whether to load the entire dataset into
              memory.
            transform: A function/transform that takes in a PIL image
              and returns a transformed version.
            target_transform: A function/transform that takes in the
              target and transforms it.
        """

        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        self.split = "train" if train else "test"
        self.split_dir = Path(self.processed_folder) / self.split

        self.logger = logging.getLogger(__name__)

        if not self._is_downloaded():
            self._download()

        if not self._is_parsed():
            self._parse_binary()

        # NOTE: Parent class provides attributes ``class_to_idx``, ``classes``, ``imgs``, and
        #       ``targets``.
        super().__init__(
            root=str(self.split_dir),
            transform=self.transform,
            target_transform=self.target_transform
        )

        # Load all images into memory, if applicable
        self.data: Optional[np.ndarray] = None
        if load_into_memory:
            self.logger.info("Loading images into memory ...")
            pbar = tqdm(
                self.imgs,
                desc="Loading FashionMNIST",
                total=len(self.imgs),
                leave=False,
                unit="image"
            )
            self.data = np.empty(
                shape=(len(self.imgs), 28, 28),
                dtype=np.uint8
            )
            for idx, (img_path, _) in enumerate(pbar):
                self.data[idx] = np.array(Image.open(img_path), dtype=np.uint8)

        # Choose appropriate loader
        self.loader = self._load_from_memory if load_into_memory else self._load_from_disk

    def __getitem__(
            self,
            index: int
    ) -> Tuple[Any, Any]:
        """Retrieve a sample from the dataset.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            A tuple (sample, target), where target is the class index of
            the target class.
        """

        sample, target = self.loader(index), self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _is_downloaded(self) -> bool:
        """Check if the dataset has been downloaded.

        Returns:
            True if the dataset has been downloaded, False otherwise.
        """

        for filename, md5 in self.raw_data:
            filepath = str(Path(self.raw_folder) / filename)
            if not check_integrity(filepath, md5):
                return False
        return True

    def _download(self) -> None:
        """Download the dataset."""

        Path(self.raw_folder).mkdir(parents=True, exist_ok=True)

        # Download files
        for filename, md5 in self.resources:
            url = f"{self.mirror}{filename}"
            download_and_extract_archive(
                url,
                download_root=self.raw_folder,
                filename=filename,
                md5=md5,
                remove_finished=True
            )

    def _is_parsed(self) -> bool:
        """Check if binary files have been parsed.

        Returns:
            True if binary files have been parsed, False otherwise.
        """

        for img_class in self.classes:
            class_dir = self.split_dir / img_class
            if not class_dir.exists() or not any(class_dir.iterdir()):
                return False
        return True

    def _parse_binary(self) -> None:
        """Parse binary files and save as PNG images."""

        # Create class subdirectories
        for img_class in self.classes:
            (self.split_dir / img_class).mkdir(parents=True, exist_ok=True)

        # Unpack raw data and save as PNG images
        image_file, label_file = self.data_by_split[self.split]
        self.logger.info(
            "Processing %s and saving images in %s ...",
            image_file,
            self.split_dir
        )
        data = read_image_file(str(Path(self.raw_folder) / image_file))
        targets = read_label_file(str(Path(self.raw_folder) / label_file))

        for idx, (img, target) in enumerate(zip(data, targets)):
            img = Image.fromarray(img.numpy(), mode="L")
            img.save(self.split_dir / self.classes[target] / f"img_{idx}.png")
        self.logger.info("All images saved successfully.")

    def _load_from_memory(
            self,
            index: int
    ) -> Image.Image:
        """Load a sample from memory."""

        return Image.fromarray(self.data[index], mode="L")

    def _load_from_disk(
            self,
            index: int
    ) -> Image.Image:
        """Load a sample from disk."""

        return default_loader(self.imgs[index][0])

    @property
    def raw_folder(self) -> str:
        """The path of the raw data folder."""

        return str(Path(self.data_dir, "raw", self.__class__.__name__))

    @property
    def processed_folder(self) -> str:
        """The path of the processed data folder."""

        return str(Path(self.data_dir, "processed", self.__class__.__name__))
