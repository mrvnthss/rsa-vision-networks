"""The ImageNet 2012 classification dataset by Deng et al. (2009)."""


import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.imagenet import load_meta_file, parse_devkit_archive
from torchvision.datasets.utils import check_integrity, extract_archive
from tqdm import tqdm


class ImageNet(ImageFolder):
    """ImageNet 2012 classification dataset (Deng et al., 2009).

    This class can be used to parse and load the ImageNet 2012
    classification dataset.  The ImageNet dataset consists of 1,331,167
    color images from 1,000 classes.  There are 1,281,167 training
    samples and 50,000 test samples.  There are between 732 and 1,300
    images per class in the training split and 50 images per class in
    the test split.

    Note:
        Prior to using this class, the ImageNet 2012 classification
        dataset has to be downloaded from the official website
        (https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)
        and placed in the "data/raw/ImageNet/" directory.  Further, as
        there are no labels provided for the test set of the ImageNet
        2012 classification dataset, we treat the validation set as the
        test set (and split the training set into training and
        validation sets) in analogy to the remaining datasets in this
        project.

    Attributes:
        class_to_idx: A dictionary mapping class names to class indices.
        classes: The class labels of the dataset, sorted alphabetically.
        data_dir: The path of the "data/" directory containing all
          datasets.
        imgs: A list of (image path, class index) tuples.
        logger: A logger instance to record logs.
        meta_data: The name of the meta file.
        raw_data: A dictionary containing the names and MD5 hashes of
          the raw data archives.
        split: The dataset split to load, either "train" or "test".
        split_dir: The directory containing the dataset split.
        target_transform: A function/transform that takes in the target
          and transforms it.
        targets: A list containing the class index for each image in the
          dataset.
        transform: A function/transform that takes in a PIL image and
          returns a transformed version.
        wnids: The WordNet IDs of the dataset classes.
        wnid_to_idx: A dictionary mapping WordNet IDs to class indices.
    """

    raw_data = {
        "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
        "test": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
        "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf")
    }

    meta_data = "meta.bin"

    def __init__(
            self,
            data_dir: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        """Initialize the ImageNet dataset.

        Args:
            data_dir: The path of the "data/" directory containing all
              datasets.
            train: Whether to load the training split (True) or the test
              split (False).
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

        if not self._is_parsed():
            self._parse_archive()

        # NOTE: Parent class provides attributes ``class_to_idx``, ``classes``, ``imgs``, and
        #       ``targets``.
        super().__init__(
            root=str(self.split_dir),
            transform=self.transform,
            target_transform=self.target_transform
        )

        # The ``classes`` attribute of the DatasetFolder class (parent class of ImageFolder) uses
        # the names of the subdirectories in the dataset ``root`` directory as class names.  For
        # ImageNet, these are the WordNet IDs of the classes.  We now replace these WordNet IDs
        # with the ImageNet class labels and store the WordNet IDs in a separate attribute.
        wnid_to_classes = load_meta_file(self.raw_folder)[0]
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

    def _is_parsed(self) -> bool:
        """Check if dataset archive has been parsed.

        Returns:
            True if the dataset archive has been parsed, False
            otherwise.
        """

        return self.split_dir.exists()

    def _parse_archive(self) -> None:
        """Parse the dataset archive and extract images."""

        # Make sure that meta file ("meta.bin") is available
        meta_fpath = str(Path(self.raw_folder) / self.meta_data)
        if not check_integrity(meta_fpath):
            self.logger.info(
                "Meta file not found in %s. Parsing %s to create meta.bin in %s ...",
                self.raw_folder,
                self.raw_data["devkit"][0],
                self.raw_folder
            )
            parse_devkit_archive(self.raw_folder, self.raw_data["devkit"][0])
            self.logger.info("Meta file created successfully.")

        if self.split == "train":
            self._parse_train_archive()
        else:
            self._parse_test_archive()

    def _verify_archive(
            self,
            file: str,
            md5: str
    ) -> None:
        """Verify the presence and integrity of an archive file.

        Args:
            file: The name of the archive file.
            md5: The MD5 hash of the archive file.

        Raises:
            RuntimeError: If the archive file is not present or is
              corrupted.
        """

        if not check_integrity(str(Path(self.raw_folder) / file), md5):
            msg = (
                "The archive {} is not present in the root directory or is corrupted. "
                "You need to download it externally and place it in {}."
            )
            raise RuntimeError(msg.format(file, self.raw_folder))

    def _parse_train_archive(self) -> None:
        """Parse the training archive and extract images."""

        filename, md5 = self.raw_data["train"]
        self.logger.info(
            "Verifying %s in %s, this may take a while ...",
            filename,
            self.raw_folder
        )
        self._verify_archive(filename, md5)
        self.logger.info("Verification successful.")

        train_archive = str(Path(self.raw_folder) / filename)
        train_root = str(Path(self.processed_folder) / "train")
        self.logger.info(
            "Extracting %s to %s, this may take a while ...",
            train_archive,
            train_root
        )
        extract_archive(train_archive, train_root)
        self.logger.info("Archive extracted successfully.")

        self.logger.info(
            "Extracting archives in %s, this may take a while ...",
            train_root
        )
        desc = "Extracting archives"
        total_archives = sum(1 for _ in Path(train_root).iterdir())
        pbar = tqdm(
            Path(train_root).iterdir(),
            desc=desc,
            total=total_archives,
            unit="archive"
        )
        for archive in pbar:
            extract_archive(str(archive), str(archive.with_suffix('')), remove_finished=True)
        self.logger.info("Archives extracted successfully.")

    def _parse_test_archive(self) -> None:
        """Parse the test archive and extract images."""

        filename, md5 = self.raw_data["test"]
        self.logger.info(
            "Verifying %s in %s ...",
            filename,
            self.raw_folder
        )
        self._verify_archive(filename, md5)
        self.logger.info("Verification successful.")

        test_archive = str(Path(self.raw_folder) / filename)
        test_root = str(Path(self.processed_folder) / "test")
        self.logger.info(
            "Extracting %s to %s ...",
            test_archive,
            test_root
        )
        extract_archive(test_archive, test_root)
        self.logger.info("Archive extracted successfully.")

        wnids = load_meta_file(self.raw_folder)[1]
        images = sorted(str(image) for image in Path(test_root).iterdir())

        # Create class subdirectories
        for wnid in set(wnids):
            Path(test_root, wnid).mkdir(parents=True, exist_ok=True)

        # Move images to appropriate class subdirectories
        self.logger.info(
            "Moving images to class subdirectories in %s ...",
            test_root
        )
        desc = "Moving images"
        total_images = len(images)
        pbar = tqdm(
            zip(wnids, images),
            desc=desc,
            total=total_images,
            unit="image"
        )
        for wnid, image in pbar:
            shutil.move(image, Path(test_root) / wnid / Path(image).name)
        self.logger.info("All images moved successfully.")

    @property
    def raw_folder(self) -> str:
        """The path of the raw data folder."""

        return str(Path(self.data_dir, "raw", self.__class__.__name__))

    @property
    def processed_folder(self) -> str:
        """The path of the processed data folder."""

        return str(Path(self.data_dir, "processed", self.__class__.__name__))
