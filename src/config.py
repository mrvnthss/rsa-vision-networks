"""Configuration file to be used by the Hydra framework."""


from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, MISSING


class DatasetName(Enum):
    CIFAR10 = "CIFAR10"
    FashionMNIST = "FashionMNIST"
    ImageNet = "ImageNet"


class ModelName(Enum):
    LeNet = "LeNet"
    VGG = "VGG"


class VGGNumLayers(Enum):
    L11 = 11
    L13 = 13
    L16 = 16
    L19 = 19


class OptimizerName(Enum):
    Adam = "Adam"
    SGD = "SGD"


class MainSchedulerName(Enum):
    CosineAnnealingLR = "CosineAnnealingLR"
    ExponentialLR = "ExponentialLR"
    StepLR = "StepLR"


class WarmupSchedulerName(Enum):
    ConstantLR = "ConstantLR"
    LinearLR = "LinearLR"


class ComputeRDMName(Enum):
    correlation = "correlation"
    euclidean = "euclidean"


class CompareRDMName(Enum):
    cosine = "cosine"


class RSATransformName(Enum):
    abs = "abs"
    square = "square"


@dataclass
class CriterionConf:
    _target_: str = MISSING


@dataclass
class CropScaleConf:
    lower: float = 1.0
    upper: float = 1.0


@dataclass
class CropRatioConf:
    lower: float = 1.0
    upper: float = 1.0


@dataclass
class TransformTrainConf:
    mean: List[float] = MISSING
    std: List[float] = MISSING
    crop_size: int = MISSING
    crop_scale: CropScaleConf = MISSING
    crop_ratio: CropRatioConf = MISSING
    flip_prob: float = 0.0
    ta_wide: bool = False
    random_erase_prob: float = 0.0


@dataclass
class TransformValConf:
    mean: List[float] = MISSING
    std: List[float] = MISSING
    resize_size: int = MISSING
    crop_size: int = MISSING


@dataclass
class VisionDatasetConf:
    _target_: str = MISSING
    data_dir: str = MISSING
    train: bool = MISSING
    load_into_memory: bool = MISSING


@dataclass
class DatasetConf:
    name: DatasetName = MISSING
    num_classes: int = MISSING
    is_grayscale: bool = MISSING
    transform_train: TransformTrainConf = MISSING
    transform_val: TransformValConf = MISSING
    train_set: VisionDatasetConf = MISSING
    test_set: VisionDatasetConf = MISSING


@dataclass
class LeNetConf:
    _target_: str = "models.lenet.LeNet"
    num_classes: int = MISSING


@dataclass
class VGGConf:
    _target_: str = "models.vgg.VGG"
    num_layers: VGGNumLayers = MISSING
    num_classes: int = MISSING
    pretrained: bool = False


@dataclass
class ModelConf:
    name: ModelName = MISSING
    kwargs: Union[LeNetConf, VGGConf] = MISSING
    input_size: int = MISSING
    load_weights_from: Optional[str] = None


@dataclass
class AdamConf:
    _target_: str = "torch.optim.adam.Adam"
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
    name: OptimizerName = MISSING
    kwargs: Union[AdamConf, SGDConf] = MISSING


@dataclass
class ConstantLRConf:
    _target_: str = "torch.optim.lr_scheduler.ConstantLR"
    factor: float = 0.3333333333333333
    total_iters: int = 5
    last_epoch: int = -1


@dataclass
class CosineAnnealingLRConf:
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    T_max: int = MISSING
    eta_min: float = 0
    last_epoch: int = -1


@dataclass
class ExponentialLRConf:
    _target_: str = "torch.optim.lr_scheduler.ExponentialLR"
    gamma: float = MISSING
    last_epoch: int = -1


@dataclass
class LinearLRConf:
    _target_: str = "torch.optim.lr_scheduler.LinearLR"
    start_factor: float = 0.3333333333333333
    end_factor: float = 1.0
    total_iters: int = 5
    last_epoch: int = -1


@dataclass
class StepLRConf:
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    step_size: int = MISSING
    gamma: float = 0.1
    last_epoch: int = -1


@dataclass
class MainSchedulerConf:
    name: MainSchedulerName = MISSING
    kwargs: Union[
        CosineAnnealingLRConf,
        ExponentialLRConf,
        StepLRConf
    ] = MISSING


@dataclass
class WarmupSchedulerConf:
    name: WarmupSchedulerName = MISSING
    warmup_epochs: int = MISSING
    kwargs: Union[
        ConstantLRConf,
        LinearLRConf
    ] = MISSING


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
    val_split: float = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING


@dataclass
class TrainingConf:
    num_epochs: int = MISSING
    resume_from: Optional[str] = None
    num_folds: Optional[int] = None


@dataclass
class MulticlassAccuracyConf:
    _target_: str = "torchmetrics.classification.MulticlassAccuracy"
    num_classes: int = MISSING
    top_k: int = 1
    average: str = "macro"
    multidim_average: str = "global"
    ignore_index: Optional[int] = None
    validate_args: bool = True


@dataclass
class PerformanceConf:
    metric: str = MISSING
    higher_is_better: bool = MISSING
    evaluate_on: str = MISSING
    patience: Optional[int] = None
    keep_previous_best_score: bool = MISSING


@dataclass
class CheckpointsConf:
    save_frequency: Optional[int] = None
    save_best_model: bool = MISSING
    delete_previous: bool = MISSING


@dataclass
class NumUpdatesConf:
    Train: Optional[int] = 10
    Val: Optional[int] = 10


@dataclass
class TensorBoardConf:
    updates_per_epoch: NumUpdatesConf = MISSING


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
    dataloader: DataloaderConf = MISSING
    evaluate_on: str = MISSING
    metrics: Dict[str, Union[MulticlassAccuracyConf]] = MISSING


@dataclass
class TrainClassifierConf(DictConfig):
    criterion: CriterionConf = MISSING
    dataset: DatasetConf = MISSING
    model: ModelConf = MISSING
    optimizer: OptimizerConf = MISSING
    main_scheduler: Optional[MainSchedulerConf] = None
    warmup_scheduler: Optional[WarmupSchedulerConf] = None
    experiment: ExperimentConf = MISSING
    paths: PathsConf = MISSING
    reproducibility: ReproducibilityConf = MISSING
    dataloader: DataloaderConf = MISSING
    training: TrainingConf = MISSING
    metrics: Dict[str, Union[MulticlassAccuracyConf]] = MISSING
    performance: PerformanceConf = MISSING
    checkpoints: CheckpointsConf = MISSING
    tensorboard: TensorBoardConf = MISSING


@dataclass
class ComputeRDMConf:
    name: ComputeRDMName = MISSING
    kwargs: Dict[str, Any] = MISSING


@dataclass
class CompareRDMConf:
    name: CompareRDMName = MISSING
    kwargs: Dict[str, Any] = MISSING


@dataclass
class ReprSimilarityConf:
    compute_rdm: ComputeRDMConf = MISSING
    compare_rdm: CompareRDMConf = MISSING
    weight_rsa_score: float = MISSING
    rsa_transform: RSATransformName = None


@dataclass
class HooksConf:
    train: str = MISSING
    ref: str = MISSING


@dataclass
class TrainSimilarityConf(DictConfig):
    dataset: DatasetConf = MISSING
    model: ModelConf = MISSING
    optimizer: OptimizerConf = MISSING
    main_scheduler: Optional[MainSchedulerConf] = None
    warmup_scheduler: Optional[WarmupSchedulerConf] = None
    repr_similarity: ReprSimilarityConf = MISSING
    experiment: ExperimentConf = MISSING
    paths: PathsConf = MISSING
    reproducibility: ReproducibilityConf = MISSING
    dataloader: DataloaderConf = MISSING
    training: TrainingConf = MISSING
    metrics: Dict[str, Union[MulticlassAccuracyConf]] = MISSING
    performance: PerformanceConf = MISSING
    checkpoints: CheckpointsConf = MISSING
    tensorboard: TensorBoardConf = MISSING
    hooks: HooksConf = MISSING
