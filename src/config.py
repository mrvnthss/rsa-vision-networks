"""Configuration file to be used by the Hydra framework."""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from omegaconf import DictConfig, MISSING
from typing_extensions import Literal


@dataclass
class CriterionConf:
    _target_: str = MISSING


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
    test_set: VisionDatasetConf = MISSING


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
class AdamConf:
    _target_: str = "torch.optim.adam.Adam"
    params: Any = MISSING
    lr: float = 1e-3
    betas: List[float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False
    foreach: Optional[bool] = None
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False
    fused: Optional[bool] = None


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
class OptimizerConf:
    type: str = MISSING
    adam: AdamConf = MISSING
    sgd: SGDConf = MISSING


@dataclass
class PathsConf:
    checkpoints: str = MISSING
    data: str = MISSING
    logs: str = MISSING
    tensorboard: str = MISSING


@dataclass
class ExperimentConf:
    name: str = MISSING
    subdir: str = MISSING


@dataclass
class SeedsConf:
    torch: int = MISSING
    shuffle: int = MISSING
    split: int = MISSING


@dataclass
class DataloaderConf:
    which_split: Literal["Train", "Val", "Test"] = MISSING
    val_split: float = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING


@dataclass
class TrainingConf:
    num_epochs: int = MISSING
    resume_from: Optional[str] = None


@dataclass
class MetricConf:
    _target_: str = MISSING
    params: Dict[str, Any] = MISSING


@dataclass
class PerformanceConf:
    metric: str = MISSING
    higher_is_better: bool = MISSING
    dataset: Literal["train", "val"] = MISSING
    patience: Optional[int] = None
    keep_previous_best_score: bool = MISSING


@dataclass
class CheckpointsConf:
    dir: str = MISSING
    save_frequency: Optional[int] = None
    save_best_model: bool = MISSING
    delete_previous: bool = MISSING


@dataclass
class TensorBoardConf:
    dir: str = MISSING
    updates_per_epoch: Dict[Literal['Train', 'Val'], Optional[int]] = MISSING


@dataclass
class ComputeStatsConf(DictConfig):
    dataset: DatasetConf = MISSING
    paths: PathsConf = MISSING


@dataclass
class TestClassifierConf(DictConfig):
    criterion: CriterionConf = MISSING
    dataset: DatasetConf = MISSING
    model: ModelConf = MISSING
    paths: PathsConf = MISSING
    seeds: SeedsConf = MISSING
    model_checkpoint: str = MISSING
    dataloader: DataloaderConf = MISSING


@dataclass
class TrainClassifierConf(DictConfig):
    criterion: CriterionConf = MISSING
    dataset: DatasetConf = MISSING
    model: ModelConf = MISSING
    optimizer: OptimizerConf = MISSING
    experiment: ExperimentConf = MISSING
    paths: PathsConf = MISSING
    seeds: SeedsConf = MISSING
    dataloader: DataloaderConf = MISSING
    training: TrainingConf = MISSING
    metrics: Dict[str, MetricConf] = MISSING
    performance: PerformanceConf = MISSING
    checkpoints: CheckpointsConf = MISSING
    tensorboard: TensorBoardConf = MISSING
