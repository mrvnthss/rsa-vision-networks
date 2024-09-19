"""Configuration file to be used by the Hydra framework."""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import torch
from omegaconf import DictConfig, MISSING


@dataclass
class CriterionConf:
    _target_: str = MISSING


@dataclass
class TransformConf:
    mean: List[float] = MISSING
    std: List[float] = MISSING
    crop_size: int = MISSING
    crop_scale: Dict[Literal["lower", "upper"], float] = MISSING
    crop_ratio: Dict[Literal["lower", "upper"], float] = MISSING
    resize_size: int = MISSING
    flip_prob: float = MISSING


@dataclass
class VisionDatasetConf:
    root: Union[str, Path] = MISSING
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None


@dataclass
class DatasetConf:
    name: str = MISSING
    num_classes: int = MISSING
    is_grayscale: bool = MISSING
    transform_params: TransformConf = MISSING
    train_set: VisionDatasetConf = MISSING
    test_set: VisionDatasetConf = MISSING


@dataclass
class ArchConf:
    _target_: str = MISSING
    num_layers: int = MISSING
    num_classes: int = MISSING
    pretrained: bool = MISSING


@dataclass
class ModelConf:
    name: str = MISSING
    architecture: ArchConf = MISSING
    input_size: int = MISSING
    load_weights_from: Optional[str] = None


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
    name: Literal["Adam", "SGD"] = MISSING
    kwargs: Union[AdamConf, SGDConf] = MISSING


@dataclass
class ExponentialLRConf:
    _target_: str = "torch.optim.lr_scheduler.ExponentialLR",
    gamma: float = MISSING,
    last_epoch: int = -1


@dataclass
class StepLRConf:
    _target_: str = "torch.optim.lr_scheduler.StepLR",
    step_size: int = MISSING,
    gamma: float = 0.1,
    last_epoch: int = -1


@dataclass
class LRSchedulerConf:
    name: Literal["ExponentialLR", "StepLR"] = MISSING,
    kwargs: Union[ExponentialLRConf, StepLRConf] = MISSING


@dataclass
class PathsConf:
    checkpoints: str = MISSING
    data: str = MISSING
    tensorboard: str = MISSING


@dataclass
class ExperimentConf:
    name: str = MISSING
    dir: str = MISSING
    sub_dir: str = MISSING


@dataclass
class ReproducibilityConf:
    torch_seed: int = MISSING
    shuffle_seed: int = MISSING
    split_seed: int = MISSING
    cudnn_deterministic: bool = MISSING
    cudnn_benchmark: bool = MISSING


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
    num_folds: Optional[int] = None


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
    save_frequency: Optional[int] = None
    save_best_model: bool = MISSING
    delete_previous: bool = MISSING


@dataclass
class TensorBoardConf:
    updates_per_epoch: Dict[Literal["Train", "Val"], Optional[int]] = MISSING


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
    reproducibility: ReproducibilityConf = MISSING
    model_checkpoint: str = MISSING
    dataloader: DataloaderConf = MISSING
    metrics: Dict[str, MetricConf] = MISSING


@dataclass
class TrainClassifierConf(DictConfig):
    criterion: CriterionConf = MISSING
    dataset: DatasetConf = MISSING
    model: ModelConf = MISSING
    optimizer: OptimizerConf = MISSING
    lr_scheduler: Optional[LRSchedulerConf] = None
    experiment: ExperimentConf = MISSING
    paths: PathsConf = MISSING
    reproducibility: ReproducibilityConf = MISSING
    dataloader: DataloaderConf = MISSING
    training: TrainingConf = MISSING
    metrics: Dict[str, MetricConf] = MISSING
    performance: PerformanceConf = MISSING
    checkpoints: CheckpointsConf = MISSING
    tensorboard: TensorBoardConf = MISSING


@dataclass
class RDMComputeConf(DictConfig):
    name: Literal["correlation", "euclidean"] = MISSING
    kwargs: Dict[str, Any] = MISSING


@dataclass
class RDMCompareConf(DictConfig):
    name: Literal["cosine"] = MISSING
    kwargs: Dict[str, Any] = MISSING


@dataclass
class RDMConf(DictConfig):
    compute: RDMComputeConf = MISSING
    compare: RDMCompareConf = MISSING


@dataclass
class HooksConf(DictConfig):
    train: str = MISSING
    ref: str = MISSING


@dataclass
class ReprSimilarityConf(DictConfig):
    weight_rsa_score: float = MISSING
    rsa_transform: Optional[Literal["abs", "square"]] = None


@dataclass
class TrainSimilarityConf(DictConfig):
    dataset: DatasetConf = MISSING
    model: ModelConf = MISSING
    optimizer: OptimizerConf = MISSING
    lr_scheduler: Optional[LRSchedulerConf] = None
    rdm: RDMConf = MISSING
    experiment: ExperimentConf = MISSING
    paths: PathsConf = MISSING
    reproducibility: ReproducibilityConf = MISSING
    dataloader: DataloaderConf = MISSING
    training: TrainingConf = MISSING
    metrics: Dict[str, MetricConf] = MISSING
    performance: PerformanceConf = MISSING
    checkpoints: CheckpointsConf = MISSING
    tensorboard: TensorBoardConf = MISSING
    hooks: HooksConf = MISSING
    repr_similarity: ReprSimilarityConf = MISSING
