from pathlib import Path
import pickle
import shutil
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class CIFAR10(VisionDataset):
    """CIFAR10 Dataset."""

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
            download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to "
                               "download it")

        self.data, self.targets = self._load_data()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        # Convert to PIL Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> None:
        if self._check_exists():
            return

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
        for file in ["batches.meta", "readme.html"]:
            file_path = Path(self.raw_folder, file)
            if file_path.exists():
                file_path.unlink()

    def _check_exists(self) -> bool:
        for filename, md5 in self.train_batches + self.test_batches:
            filepath = str(Path(self.raw_folder) / filename)
            if not check_integrity(filepath, md5):
                return False
        return True

    def _load_data(self):
        data = []
        targets = []

        batches = self.train_batches if self.train else self.test_batches

        for filename, md5 in batches:
            filepath = str(Path(self.raw_folder) / filename)
            with open(filepath, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                data.append(entry["data"])
                targets.extend(entry["labels"])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))

        return data, targets

    @property
    def raw_folder(self) -> str:
        return str(Path(self.root, "raw", self.__class__.__name__))

    @property
    def processed_folder(self) -> str:
        return str(Path(self.root, "processed", self.__class__.__name__))

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}
