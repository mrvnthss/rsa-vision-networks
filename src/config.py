"""Configuration file to be used by the Hydra framework."""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from omegaconf import DictConfig, MISSING


@dataclass
class ArchConf:
    _target_: str = MISSING
    num_layers: int = MISSING
    num_classes: int = MISSING
    pretrained: bool = MISSING


@dataclass
class ComposeConf:
    _target_: str = "torchvision.transforms.v2.Compose"
    transforms: Any = MISSING


@dataclass
class ModelConf:
    name: str = MISSING
    architecture: ArchConf = MISSING
    preprocessing: ComposeConf = MISSING


@dataclass
class VisionDatasetConf:
    root: Union[str, Path] = MISSING
    transforms: Optional[Callable] = None
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None


@dataclass
class DatasetConf:
    name: str = MISSING
    num_classes: int = MISSING
    is_grayscale: bool = MISSING
    norm_constants: Dict[str, List[float]] = MISSING
    train_set: VisionDatasetConf = MISSING
    val_set: VisionDatasetConf = MISSING


@dataclass
class LossConf:
    _target_: str = MISSING


@dataclass
class SGDConf:
    _target_: str = "torch.optim.sgd.SGD"
    params: Any = MISSING
    lr: float = 1e-3
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False
    maximize: bool = False
    foreach: Optional[bool] = None
    differentiable: bool = False
    fused: Optional[bool] = None


@dataclass
class PathsConf:
    data: str = MISSING
    logs: str = MISSING


@dataclass
class DataloaderConf:
    batch_size: int = MISSING
    num_workers: int = MISSING


@dataclass
class TrainingConf:
    seed: int = MISSING
    resume_from: str = MISSING
    num_epochs: int = MISSING
    tb_updates: int = MISSING
    performance_metric: str = MISSING


@dataclass
class CheckpointsConf:
    checkpoint_dir: str = MISSING
    save_frequency: int = MISSING
    delete_previous: bool = MISSING
    save_best: bool = MISSING
    patience: int = MISSING


@dataclass
class ClassifierConf(DictConfig):
    model: ModelConf
    dataset: DatasetConf
    loss: LossConf
    optimizer: SGDConf
    run: str
    paths: PathsConf
    dataloader: DataloaderConf
    training: TrainingConf
    checkpoints: CheckpointsConf


@dataclass
class ComputeStatsConf(DictConfig):
    dataset: DatasetConf
    paths: PathsConf
