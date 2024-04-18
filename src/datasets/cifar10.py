"""The CIFAR10 dataset by Krizhevsky (2009)."""


from pathlib import Path
import pickle
import shutil
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

    Params:
        root: Root directory of the dataset.
        train: If True, loads the training split, else the test split.
        transform: A transform to modify features (images).
        target_transform: A transform to modify targets (labels).

    (Additional) Attributes:
        split: The dataset split to load, either "train" or "val".
        split_dir: Directory containing the dataset split.
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
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.split = "train" if train else "val"
        self.split_dir = Path(self.processed_folder) / self.split
        self.transform = transform
        self.target_transform = target_transform

        if not self._is_downloaded():
            self._download()

        if not self._is_parsed():
            self._parse_binary()

        super().__init__(
            str(self.split_dir),
            transform=self.transform,
            target_transform=self.target_transform
        )

        # NOTE: Calling the super constructor leads to the self.root attribute being overwritten
        #       and set to the directory containing the processed data (i.e., the images arranged
        #       in subdirectories by class). Here, we reset it to the original root directory.
        self.root = root

    def _is_downloaded(self) -> bool:
        for filename, md5 in self.train_batches + self.test_batches:
            filepath = str(Path(self.raw_folder) / filename)
            if not check_integrity(filepath, md5):
                return False
        return True

    def _download(self) -> None:
        Path(self.raw_folder).mkdir(parents=True, exist_ok=True)

        # Download files
        filename, md5 = self.resource
        url = f"{self.mirror}{filename}"
        download_and_extract_archive(
            url, download_root=self.raw_folder, extract_root=self.raw_folder,
            filename=filename, md5=md5, remove_finished=True
        )
        print()

        # Remove intermediate "cifar-10-batches-py" directory
        intermediate_dir = Path(self.raw_folder, "cifar-10-batches-py")
        for file in intermediate_dir.iterdir():
            shutil.move(str(file), self.raw_folder)
        intermediate_dir.rmdir()

        # Delete auxiliary files
        for filename in ["batches.meta", "readme.html"]:
            filepath = Path(self.raw_folder, filename)
            if filepath.exists():
                filepath.unlink()

    def _is_parsed(self) -> bool:
        for img_class in self.classes:
            class_dir = self.split_dir / img_class
            if not class_dir.exists() or not any(class_dir.iterdir()):
                return False
        return True

    def _parse_binary(self) -> None:
        # Create subdirectories for each class
        for img_class in self.classes:
            (self.split_dir / img_class).mkdir(parents=True, exist_ok=True)

        # Unpack raw data and save as PNG images
        batches = self.train_batches if self.split == "train" else self.test_batches
        for filename, _ in batches:
            filepath = str(Path(self.raw_folder) / filename)
            with open(filepath, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                data = np.array(entry["data"]).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
                targets = entry["labels"]

            for idx, (img, target) in enumerate(zip(data, targets)):
                img = Image.fromarray(img, mode="RGB")
                img.save(self.split_dir / self.classes[target] / f"img_{idx}.png")

    @property
    def raw_folder(self) -> str:
        return str(Path(self.root, "raw", self.__class__.__name__))

    @property
    def processed_folder(self) -> str:
        return str(Path(self.root, "processed", self.__class__.__name__))
