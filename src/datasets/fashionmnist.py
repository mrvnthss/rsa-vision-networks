from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class FashionMNIST(VisionDataset):
    """FashionMNIST Dataset by Zalando Research."""

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

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ]

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False
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

        # Convert PyTorch tensor to PIL Image (8-bit pixels, grayscale)
        img = Image.fromarray(img.numpy(), mode="L")

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
        for filename, md5 in self.resources:
            url = f"{self.mirror}{filename}"
            download_and_extract_archive(
                url, download_root=self.raw_folder, filename=filename, md5=md5,
                remove_finished=True
            )
            print()

    def _check_exists(self) -> bool:
        for filename, md5 in self.raw_data:
            filepath = str(Path(self.raw_folder) / filename)
            if not check_integrity(filepath, md5):
                return False
        return True

    def _load_data(self):
        prefix = "train" if self.train else "t10k"
        image_file = f"{prefix}-images-idx3-ubyte"
        data = read_image_file(str(Path(self.raw_folder, image_file)))

        label_file = f"{prefix}-labels-idx1-ubyte"
        targets = read_label_file(str(Path(self.raw_folder, label_file)))

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
