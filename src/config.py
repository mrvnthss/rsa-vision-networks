# pylint: disable=invalid-name,missing-class-docstring

"""Configuration file to be used by the Hydra framework."""


from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, MISSING


# -------------
# MODEL CHOICES
# -------------

class ModelName(Enum):
    LeNet = "LeNet"
    VGG = "VGG"


class VGGNumLayers(Enum):
    L11 = 11
    L13 = 13
    L16 = 16
    L19 = 19


# ---------------
# DATASET CHOICES
# ---------------

class DatasetName(Enum):
    CIFAR10 = "CIFAR10"
    FashionMNIST = "FashionMNIST"
    ImageNet = "ImageNet"


class DatasetSplitName(Enum):
    train = "train"
    val = "val"
    test = "test"


# -----------------
# CRITERION CHOICES
# -----------------

class CriterionName(Enum):
    CrossEntropyLoss = "CrossEntropyLoss"


# -----------
# RSA CHOICES
# -----------

class ComputeRDMName(Enum):
    correlation = "correlation"
    euclidean = "euclidean"


class CompareRDMName(Enum):
    correlation = "correlation"
    cosine = "cosine"
    spearman = "spearman"


class RSATransformName(Enum):
    abs = "abs"
    square = "square"


# -----------------
# OPTIMIZER CHOICES
# -----------------

class OptimizerName(Enum):
    Adam = "Adam"
    SGD = "SGD"


# --------------------
# LR SCHEDULER CHOICES
# --------------------

class MainSchedulerName(Enum):
    CosineAnnealingLR = "CosineAnnealingLR"
    ExponentialLR = "ExponentialLR"
    StepLR = "StepLR"


class WarmupSchedulerName(Enum):
    ConstantLR = "ConstantLR"
    LinearLR = "LinearLR"


# --------------------
# MODEL CONFIGURATIONS
# --------------------

@dataclass
class LeNetConf:
    _target_: str = "models.lenet.LeNet"
    num_classes: int = MISSING


@dataclass
class VGGConf:
    _target_: str = "models.vgg.VGG"
    num_layers: VGGNumLayers = MISSING
    num_classes: int = MISSING
    pretrained: bool = MISSING


@dataclass
class ModelConf:
    name: ModelName = MISSING
    input_size: int = MISSING
    num_layers: Optional[int] = None
    pretrained: Optional[bool] = None
    load_weights_from: Optional[str] = None
    evaluate_on: Optional[DatasetSplitName] = None
    kwargs: Union[
        LeNetConf,
        VGGConf
    ] = MISSING


# ----------------------
# DATASET CONFIGURATIONS
# ----------------------

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
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0


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
    load_into_memory: Optional[bool] = None


@dataclass
class DatasetConf:
    name: DatasetName = MISSING
    num_classes: int = MISSING
    is_grayscale: bool = MISSING
    transform_train: TransformTrainConf = MISSING
    transform_val: TransformValConf = MISSING
    train_set: VisionDatasetConf = MISSING
    test_set: VisionDatasetConf = MISSING


# ------------------------
# CRITERION CONFIGURATIONS
# ------------------------

@dataclass
class CrossEntropyLossConf:
    _target_: str = "torch.nn.CrossEntropyLoss"
    label_smoothing: float = 0.0

@dataclass
class CriterionConf:
    name: CriterionName = MISSING
    label_smoothing: float = 0.0
    kwargs: Union[
        CrossEntropyLossConf
    ] = MISSING


# ------------------
# RSA CONFIGURATIONS
# ------------------

@dataclass
class HooksConf:
    train: str = MISSING
    ref: str = MISSING


@dataclass
class ComputeRDMCorrelationConf:
    pass


@dataclass
class ComputeRDMEuclideanConf:
    center_activations: bool = False
    normalize_distances: bool = True
    distance_type: Optional[str] = "squared"


@dataclass
class ComputeRDMConf:
    name: ComputeRDMName = MISSING
    center_activations: Optional[bool] = None
    normalize_distances: Optional[bool] = None
    distance_type: Optional[str] = None
    kwargs: Union[
        ComputeRDMCorrelationConf,
        ComputeRDMEuclideanConf
    ] = MISSING


@dataclass
class CompareRDMCorrelationConf:
    pass


@dataclass
class CompareRDMCosineConf:
    pass


@dataclass
class CompareRDMSpearmanConf:
    pass


@dataclass
class CompareRDMConf:
    name: CompareRDMName = MISSING
    kwargs: Union[
        CompareRDMCorrelationConf,
        CompareRDMCosineConf,
        CompareRDMSpearmanConf
    ] = MISSING


@dataclass
class ReprSimilarityConf:
    hooks: HooksConf = MISSING
    compute_rdm: ComputeRDMConf = MISSING
    compare_rdm: CompareRDMConf = MISSING
    weight_rsa_score: float = MISSING
    rsa_transform: RSATransformName = None


# ------------------------
# OPTIMIZER CONFIGURATIONS
# ------------------------

@dataclass
class AdamConf:
    _target_: str = "torch.optim.adam.Adam"
    lr: float = MISSING
    betas: Tuple[float] = (0.9, 0.999)
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
    lr: float = MISSING
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
    lr: float = MISSING
    betas: Optional[Tuple[float]] = None
    momentum: Optional[float] = None
    dampening: Optional[float] = None
    weight_decay: float = 0
    kwargs: Union[
        AdamConf,
        SGDConf
    ] = MISSING


# -----------------------------
# MAIN SCHEDULER CONFIGURATIONS
# -----------------------------

@dataclass
class CosineAnnealingLRConf:
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    T_max: int = MISSING
    eta_min: float = MISSING
    last_epoch: int = -1


@dataclass
class ExponentialLRConf:
    _target_: str = "torch.optim.lr_scheduler.ExponentialLR"
    gamma: float = MISSING
    last_epoch: int = -1


@dataclass
class StepLRConf:
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    step_size: int = MISSING
    gamma: float = MISSING
    last_epoch: int = -1


@dataclass
class MainSchedulerConf:
    name: MainSchedulerName = MISSING
    lr_min: Optional[float] = None
    lr_step_size: Optional[float] = None
    lr_gamma: Optional[float] = None
    kwargs: Union[
        CosineAnnealingLRConf,
        ExponentialLRConf,
        StepLRConf
    ] = MISSING


# -------------------------------
# WARMUP SCHEDULER CONFIGURATIONS
# -------------------------------

@dataclass
class ConstantLRConf:
    _target_: str = "torch.optim.lr_scheduler.ConstantLR"
    factor: float = MISSING
    total_iters: int = MISSING
    last_epoch: int = -1


@dataclass
class LinearLRConf:
    _target_: str = "torch.optim.lr_scheduler.LinearLR"
    start_factor: float = MISSING
    end_factor: float = 1.0
    total_iters: int = MISSING
    last_epoch: int = -1


@dataclass
class WarmupSchedulerConf:
    name: WarmupSchedulerName = MISSING
    warmup_decay: float = MISSING
    warmup_epochs: int = MISSING
    kwargs: Union[
        ConstantLRConf,
        LinearLRConf
    ] = MISSING


# -------------------------
# EXPERIMENT CONFIGURATIONS
# -------------------------

@dataclass
class ExperimentConf:
    name: str = MISSING
    dir: str = MISSING
    sub_dir: Optional[str] = None


# ------------------------------
# REPRODUCIBILITY CONFIGURATIONS
# ------------------------------

@dataclass
class ReproducibilityConf:
    torch_seed: int = MISSING
    shuffle_seed: int = MISSING
    split_seed: int = MISSING
    cudnn_deterministic: bool = MISSING
    cudnn_benchmark: bool = MISSING


# -----------------------
# TRAINING CONFIGURATIONS
# -----------------------

@dataclass
class TrainingConf:
    num_epochs: int = MISSING
    num_folds: Optional[int] = None
    resume_from: Optional[str] = None


# -------------------------
# DATALOADER CONFIGURATIONS
# -------------------------

@dataclass
class DataloaderConf:
    val_split: Optional[float] = None
    batch_size: int = MISSING
    num_workers: int = MISSING


# ----------------------------------
# PERFORMANCE METRICS CONFIGURATIONS
# ----------------------------------

@dataclass
class MulticlassAccuracyConf:
    _target_: str = "torchmetrics.classification.MulticlassAccuracy"
    num_classes: int = MISSING
    top_k: int = MISSING
    average: str = "micro"
    multidim_average: str = "global"
    ignore_index: Optional[int] = None
    validate_args: bool = True


@dataclass
class PerformanceConf:
    metric: str = MISSING
    higher_is_better: bool = MISSING
    evaluate_on: DatasetSplitName = MISSING
    patience: Optional[int] = None
    keep_previous_best_score: bool = MISSING


# -------------------------------
# LOGGING & SAVING CONFIGURATIONS
# -------------------------------

@dataclass
class CheckpointsConf:
    save_frequency: Optional[int] = None
    save_best_model: bool = MISSING
    delete_previous: bool = MISSING


@dataclass
class NumUpdatesConf:
    train: Optional[int] = None
    val: Optional[int] = None


@dataclass
class TensorBoardConf:
    updates_per_epoch: NumUpdatesConf = MISSING


@dataclass
class PathsConf:
    checkpoints: str = MISSING
    tensorboard: str = MISSING


# ------------------
# EXECUTABLE SCRIPTS
# ------------------

@dataclass
class ComputeStatsConf(DictConfig):
    dataset: DatasetConf = MISSING


@dataclass
class TestClassifierConf(DictConfig):
    model: ModelConf = MISSING
    dataset: DatasetConf = MISSING
    dataloader: DataloaderConf = MISSING
    criterion: CriterionConf = MISSING
    reproducibility: ReproducibilityConf = MISSING
    metrics: Dict[str, Union[
        MulticlassAccuracyConf
    ]] = MISSING


@dataclass
class TrainClassifierConf(DictConfig):
    model: ModelConf = MISSING
    dataset: DatasetConf = MISSING
    dataloader: DataloaderConf = MISSING
    criterion: CriterionConf = MISSING
    optimizer: OptimizerConf = MISSING
    main_scheduler: Optional[MainSchedulerConf] = None
    warmup_scheduler: Optional[WarmupSchedulerConf] = None
    experiment: ExperimentConf = MISSING
    reproducibility: ReproducibilityConf = MISSING
    training: TrainingConf = MISSING
    metrics: Dict[str, Union[
        MulticlassAccuracyConf
    ]] = MISSING
    performance: PerformanceConf = MISSING
    checkpoints: CheckpointsConf = MISSING
    tensorboard: TensorBoardConf = MISSING
    paths: PathsConf = MISSING


@dataclass
class TrainSimilarityConf(DictConfig):
    model: ModelConf = MISSING
    dataset: DatasetConf = MISSING
    dataloader: DataloaderConf = MISSING
    repr_similarity: ReprSimilarityConf = MISSING
    optimizer: OptimizerConf = MISSING
    main_scheduler: Optional[MainSchedulerConf] = None
    warmup_scheduler: Optional[WarmupSchedulerConf] = None
    experiment: ExperimentConf = MISSING
    reproducibility: ReproducibilityConf = MISSING
    training: TrainingConf = MISSING
    metrics: Dict[str, Union[
        MulticlassAccuracyConf
    ]] = MISSING
    performance: PerformanceConf = MISSING
    checkpoints: CheckpointsConf = MISSING
    tensorboard: TensorBoardConf = MISSING
    paths: PathsConf = MISSING
