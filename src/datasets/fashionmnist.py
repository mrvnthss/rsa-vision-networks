from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.folder import ImageFolder


class FashionMNIST(ImageFolder):
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

    mapping = {
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
        self.train = train
        self.img_folder = str(Path(self.processed_folder) / ("train" if self.train else "val"))
        self.transform = transform
        self.target_transform = target_transform

        # Download raw files, if necessary
        if not self._is_downloaded():
            self._download()

        # Process data, if necessary
        if not self._is_processed():
            self._process()

        super().__init__(
            root=self.img_folder, transform=self.transform, target_transform=self.target_transform
        )

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

    def _is_processed(self) -> bool:
        for image_set in self.mapping.keys():
            subdir = Path(self.processed_folder) / image_set
            for img_class in self.classes:
                class_dir = subdir / img_class
                if not class_dir.exists() or not any(class_dir.iterdir()):
                    return False
        return True

    def _process(self) -> None:
        # Create subdirectories for each class
        for img_class in self.classes:
            Path(self.processed_folder, "train", img_class).mkdir(parents=True, exist_ok=True)
            Path(self.processed_folder, "val", img_class).mkdir(parents=True, exist_ok=True)

        # Unpack raw data and save as PNG images
        for image_set, (image_file, label_file) in self.mapping.items():
            data = read_image_file(str(Path(self.raw_folder, image_file)))
            targets = read_label_file(str(Path(self.raw_folder, label_file)))
            subdir = Path(self.processed_folder) / image_set

            for idx, (img, target) in enumerate(zip(data, targets)):
                img = Image.fromarray(img.numpy(), mode="L")
                img.save(str(subdir / self.classes[target] / f"image_{idx}.png"))

    @property
    def raw_folder(self) -> str:
        return str(Path(self.root, "raw", self.__class__.__name__))

    @property
    def processed_folder(self) -> str:
        return str(Path(self.root, "processed", self.__class__.__name__))
