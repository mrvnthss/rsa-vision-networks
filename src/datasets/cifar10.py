"""The CIFAR10 dataset by Krizhevsky (2009)."""


import logging
import pickle
import shutil
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class CIFAR10(ImageFolder):
    """CIFAR10 dataset (Krizhevsky, 2009).

    The class can be used to download, parse, and load the CIFAR-10
    dataset.  The CIFAR-10 dataset consists of 60,000 32x32 color images
    from 10 classes, with 6,000 images per class.  There are 50,000
    training samples and 10,000 test samples.

    Attributes:
        classes: The class labels of the dataset.
        data_dir: The path of the "data/" directory containing all
          datasets.
        logger: A logger instance to record logs.
        mirror: The URL mirror to download the dataset from.
        resource: The name and MD5 hash of the dataset archive.
        split: The dataset split to load, either "train" or "test".
        split_dir: The directory containing the dataset split.
        target_transform: A transform to modify targets (labels).
        test_batches: The names and MD5 hashes of the test batch.
        train_batches: The names and MD5 hashes of the training batches.
        transform: A transform to modify features (images).
    """

    mirror = "https://www.cs.toronto.edu/~kriz/"

    resource = ("cifar-10-python.tar.gz", "c58f30108f718f92721af3b95e74349a")

    train_batches = [
        ("data_batch_1", "c99cafc152244af753f735de768cd75f"),
        ("data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"),
        ("data_batch_3", "54ebc095f3ab1f0389bbae665268c751"),
        ("data_batch_4", "634d18415352ddfa80567beed471001a"),
        ("data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb")
    ]

    test_batches = [
        ("test_batch", "40351d587109b95175f43aff81a1287e")
    ]

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    def __init__(
            self,
            data_dir: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize the CIFAR10 dataset.

        Args:
            data_dir: The path of the "data/" directory containing all
              datasets.
            train: Whether to load the training split (True) or the
              testing split (False).
            transform: A transform to modify features (images).
            target_transform: A transform to modify targets (labels).
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

        super().__init__(
            root=str(self.split_dir),
            transform=self.transform,
            target_transform=self.target_transform
        )

    def _is_downloaded(self) -> bool:
        """Check if the dataset has been downloaded.

        Returns:
            True if the dataset has been downloaded, False otherwise.
        """

        for filename, md5 in self.train_batches + self.test_batches:
            filepath = str(Path(self.raw_folder) / filename)
            if not check_integrity(filepath, md5):
                return False
        return True

    def _download(self) -> None:
        """Download the dataset."""

        Path(self.raw_folder).mkdir(parents=True, exist_ok=True)

        # Download files
        filename, md5 = self.resource
        url = f"{self.mirror}{filename}"
        download_and_extract_archive(
            url,
            download_root=self.raw_folder,
            extract_root=self.raw_folder,
            filename=filename,
            md5=md5,
            remove_finished=True
        )

        # Remove intermediate "cifar-10-batches-py" directory
        intermediate_dir = Path(self.raw_folder) / "cifar-10-batches-py"
        self.logger.info(
            "Moving all files from %s to %s ...",
            intermediate_dir,
            self.raw_folder
        )
        for file in intermediate_dir.iterdir():
            shutil.move(str(file), self.raw_folder)
        intermediate_dir.rmdir()
        self.logger.info("All files moved successfully.")

        # Delete auxiliary files
        aux_files = ["batches.meta", "readme.html"]
        self.logger.info(
            "Deleting auxiliary files %s and %s in %s ...",
            *aux_files,
            self.raw_folder
        )
        for filename in aux_files:
            filepath = Path(self.raw_folder) / filename
            if filepath.exists():
                filepath.unlink()
        self.logger.info("Auxiliary files deleted successfully.")

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
        batches = self.train_batches if self.split == "train" else self.test_batches
        img_idx = 0
        for filename, _ in batches:
            filepath = str(Path(self.raw_folder) / filename)
            self.logger.info(
                "Processing %s and saving images in %s ...",
                filepath,
                self.split_dir
            )
            with open(filepath, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                data = entry["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                targets = entry["labels"]

            for img, target in zip(data, targets):
                img = Image.fromarray(img, mode="RGB")
                img.save(self.split_dir / self.classes[target] / f"img_{img_idx}.png")
                img_idx += 1
        self.logger.info("All images saved successfully.")

    @property
    def raw_folder(self) -> str:
        """The path of the raw data folder."""

        return str(Path(self.data_dir, "raw", self.__class__.__name__))

    @property
    def processed_folder(self) -> str:
        """The path of the processed data folder."""

        return str(Path(self.data_dir, "processed", self.__class__.__name__))
