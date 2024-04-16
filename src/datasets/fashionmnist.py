"""This module provides the FashionMNIST class."""


from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class FashionMNIST(ImageFolder):
    """FashionMNIST dataset by Zalando Research (Xiao et al., 2017).

    The class can be used to download, parse, and load the FashionMNIST
    dataset.  The FashionMNIST dataset consists of 70.000 28x28
    grayscale images from 10 classes, with 7.000 images per class. There
    are 60.000 training samples and 10.000 test samples.

    Parameters:
        root: Root directory of the dataset.
        train: If True, loads the training split, else the test split.
        transform: A transform to modify features (images).
        target_transform: A transform to modify targets (labels).

    (Additional) Attributes:
        split: The dataset split to load, either "train" or "val".
        split_dir: Directory containing the dataset split.
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

    splits = {
        "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
        "val": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    }

    classes = [
        "T-Shirt",  # T-shirt/top
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Boot"  # Ankle boot
    ]

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
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
            str(self.split_dir), transform=self.transform, target_transform=self.target_transform
        )

        # NOTE: Calling the super constructor leads to the self.root attribute being overwritten
        #       and set to the directory containing the processed data (i.e., the images arranged
        #       in subdirectories by class). Here, we reset it to the original root directory.
        self.root = root

    def _is_downloaded(self) -> bool:
        for filename, md5 in self.raw_data:
            filepath = str(Path(self.raw_folder) / filename)
            if not check_integrity(filepath, md5):
                return False
        return True

    def _download(self) -> None:
        Path(self.raw_folder).mkdir(parents=True, exist_ok=True)

        # Download files
        for filename, md5 in self.resources:
            url = f"{self.mirror}{filename}"
            download_and_extract_archive(
                url, download_root=self.raw_folder, filename=filename, md5=md5,
                remove_finished=True
            )
            print()

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
        image_file, label_file = self.splits[self.split]
        data = read_image_file(str(Path(self.raw_folder, image_file)))
        targets = read_label_file(str(Path(self.raw_folder, label_file)))

        for idx, (img, target) in enumerate(zip(data, targets)):
            img = Image.fromarray(img.numpy(), mode="L")
            img.save(self.split_dir / self.classes[target] / f"img_{idx}.png")

    @property
    def raw_folder(self) -> str:
        return str(Path(self.root, "raw", self.__class__.__name__))

    @property
    def processed_folder(self) -> str:
        return str(Path(self.root, "processed", self.__class__.__name__))
