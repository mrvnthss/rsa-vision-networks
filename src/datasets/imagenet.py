"""The ImageNet 2012 classification dataset by Deng et al. (2009)."""


from pathlib import Path
import shutil
from typing import Callable, Optional

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.imagenet import load_meta_file, parse_devkit_archive
from torchvision.datasets.utils import check_integrity, extract_archive


class ImageNet(ImageFolder):
    """ImageNet 2012 classification dataset (Deng et al., 2009).

    This class can be used to parse and load the ImageNet 2012
    classification dataset.  The ImageNet dataset consists of 1,331,167
    color images from 1,000 classes.  There are 1,281,167 training
    samples and 50,000 validation samples.  There are between 732 and
    1,300 images per class in the training split and 50 images per class
    in the validation split.

    Params:
        root: Root directory of the dataset.
        train: If True, loads the training split, else the validation
          split.
        transform: A transform to modify features (images).
        target_transform: A transform to modify targets (labels).

    (Additional) Attributes:
        split: The dataset split to load, either "train" or "val".
        split_dir: Directory containing the dataset split.

    Note:
        Prior to using this class, the ImageNet 2012 classification
        dataset has to be downloaded from the official website
        (https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)
        and placed in the directory 'data/raw/ImageNet/'.
    """

    raw_data = {
        "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
        "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
        "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf")
    }

    meta_data = "meta.bin"

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.split = "train" if train else "val"
        self.split_dir = Path(self.processed_folder) / self.split

        if not self._is_parsed():
            self._parse_archive()

        # Load dictionary mapping WordNet IDs to ImageNet classes
        wnid_to_classes = load_meta_file(self.raw_folder)[0]

        super().__init__(
            str(self.split_dir),
            transform=self.transform,
            target_transform=self.target_transform
        )
        self.root = root

        # Replace WordNet IDs with ImageNet class labels
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

    def _is_parsed(self) -> bool:
        return self.split_dir.exists()

    def _parse_archive(self) -> None:
        # Make sure that meta file ("meta.bin") is available
        meta_fpath = str(Path(self.raw_folder, self.meta_data))
        if not check_integrity(meta_fpath):
            parse_devkit_archive(self.raw_folder, self.raw_data["devkit"][0])

        if self.split == "train":
            self._parse_train_archive()
        else:
            self._parse_val_archive()

    def _verify_archive(self, file: str, md5: str) -> None:
        if not check_integrity(str(Path(self.raw_folder, file)), md5):
            msg = (
                "The archive {} is not present in the root directory or is corrupted. "
                "You need to download it externally and place it in {}."
            )
            raise RuntimeError(msg.format(file, self.raw_folder))

    def _parse_train_archive(self) -> None:
        filename, md5 = self.raw_data["train"]
        self._verify_archive(filename, md5)

        train_archive = str(Path(self.raw_folder) / filename)
        train_root = str(Path(self.processed_folder) / "train")
        extract_archive(train_archive, train_root)

        for archive in Path(train_root).iterdir():
            extract_archive(str(archive), str(archive.with_suffix('')), remove_finished=True)

    def _parse_val_archive(self) -> None:
        filename, md5 = self.raw_data["val"]
        self._verify_archive(filename, md5)

        val_archive = str(Path(self.raw_folder) / filename)
        val_root = str(Path(self.processed_folder) / "val")
        extract_archive(val_archive, val_root)

        images = sorted(str(image) for image in Path(val_root).iterdir())

        wnids = load_meta_file(self.raw_folder)[1]
        for wnid in set(wnids):
            Path(val_root, wnid).mkdir(parents=True, exist_ok=True)

        for wnid, img_file in zip(wnids, images):
            shutil.move(img_file, Path(val_root, wnid, Path(img_file).name))

    @property
    def raw_folder(self) -> str:
        return str(Path(self.root, "raw", self.__class__.__name__))

    @property
    def processed_folder(self) -> str:
        return str(Path(self.root, "processed", self.__class__.__name__))
