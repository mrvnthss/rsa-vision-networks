"""The CIFAR10 dataset by Krizhevsky (2009)."""


import logging
import pickle
import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader, ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from tqdm import tqdm


class CIFAR10(ImageFolder):
    """CIFAR10 dataset (Krizhevsky, 2009).

    The class can be used to download, parse, and load the CIFAR-10
    dataset.  The CIFAR-10 dataset consists of 60,000 32x32 color images
    from 10 classes, with 6,000 images per class.  Per class, there are
    5,000 training samples and 1,000 test samples.

    Attributes:
        class_to_idx: A dictionary mapping class names to class indices.
        classes: The class labels of the dataset, sorted alphabetically.
        data: A NumPy array containing all images in the dataset if
          ``load_into_memory`` is set to True when initializing the
          dataset.
        data_dir: The path of the "data/" directory containing all
          datasets.
        imgs: A list of (image path, class index) tuples.
        loader: A function to load a sample given its index.
        logger: A logger instance to record logs.
        mirror: The URL mirror to download the dataset from.
        resource: The name and MD5 hash of the dataset archive.
        split: The dataset split to load, either "train" or "test".
        split_dir: The directory containing the dataset split.
        target_transform: A function/transform that takes in the target
          and transforms it.
        targets: A list containing the class index for each image in the
          dataset.
        test_batches: The names and MD5 hashes of the test batch.
        train_batches: The names and MD5 hashes of the training batches.
        transform: A function/transform that takes in a PIL image and
          returns a transformed version.
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
            load_into_memory: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize the CIFAR10 dataset.

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
                desc="Loading CIFAR10",
                total=len(self.imgs),
                leave=False,
                unit="image"
            )
            self.data = np.empty(
                shape=(len(self.imgs), 32, 32, 3),
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

    def _load_from_memory(
            self,
            index: int
    ) -> Image.Image:
        """Load a sample from memory."""

        return Image.fromarray(self.data[index], mode="RGB")

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
