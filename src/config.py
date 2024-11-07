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
    LeNetModified = "LeNetModified"
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
    cosine = "cosine"
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
    CosineAnnealingWarmRestarts = "CosineAnnealingWarmRestarts"
    ExponentialLR = "ExponentialLR"
    StepLR = "StepLR"


class WarmupSchedulerName(Enum):
    ConstantLR = "ConstantLR"
    LinearLR = "LinearLR"


#---------------
# OPTUNA CHOICES
#---------------

class OptunaParamType(Enum):
    categorical = "categorical"
    float = "float"
    int = "int"


class OptunaSamplerName(Enum):
    RandomSampler = "RandomSampler"
    TPESampler = "TPESampler"


class OptunaPrunerName(Enum):
    MedianPruner = "MedianPruner"
    SuccessiveHalvingPruner = "SuccessiveHalvingPruner"


# --------------------
# MODEL CONFIGURATIONS
# --------------------

@dataclass
class LeNetConf:
    _target_: str = "models.lenet.LeNet"
    num_classes: int = MISSING


@dataclass
class LeNetModifiedConf:
    _target_: str = "models.lenet_modified.LeNetModified"
    layer_widths: Tuple[int] = (6, 16, 120)
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
        LeNetModifiedConf,
        VGGConf
    ] = MISSING


# ----------------------
# DATASET CONFIGURATIONS
# ----------------------

@dataclass
class DatasetStatsConf:
    mean: List[float] = MISSING
    std: List[float] = MISSING


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
    stats: DatasetStatsConf = MISSING
    train_set: VisionDatasetConf = MISSING
    test_set: VisionDatasetConf = MISSING


# ------------------------
# TRANSFORM CONFIGURATIONS
# ------------------------

@dataclass
class CropScaleConf:
    lower: float = 0.08
    upper: float = 1.0


@dataclass
class CropRatioConf:
    lower: float = 0.75
    upper: float = 1.3333333333333333


@dataclass
class TransformTrainConf:
    mean: List[float] = MISSING
    std: List[float] = MISSING
    crop_size: Optional[int] = None
    crop_scale: Optional[CropScaleConf] = MISSING
    crop_ratio: Optional[CropRatioConf] = MISSING
    flip_prob: Optional[float] = None
    ta_wide: Optional[bool] = None
    random_erase_prob: Optional[float] = None
    mixup_alpha: Optional[float] = None
    cutmix_alpha: Optional[float] = None


@dataclass
class TransformValConf:
    mean: List[float] = MISSING
    std: List[float] = MISSING
    resize_size: int = MISSING
    crop_size: int = MISSING


@dataclass
class TransformConf:
    train: TransformTrainConf = MISSING
    val: TransformValConf = MISSING


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
class ComputeRDMCosineConf:
    pass


@dataclass
class ComputeRDMEuclideanConf:
    center_activations: bool = False
    normalize_distances: bool = True
    distance_type: str = "squared"


@dataclass
class ComputeRDMConf:
    name: ComputeRDMName = MISSING
    center_activations: Optional[bool] = None
    normalize_distances: Optional[bool] = None
    distance_type: Optional[str] = None
    kwargs: Union[
        ComputeRDMCorrelationConf,
        ComputeRDMCosineConf,
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
    differentiable: bool = False
    regularization: Optional[str] = None
    regularization_strength: Optional[float] = None


@dataclass
class CompareRDMConf:
    name: CompareRDMName = MISSING
    differentiable: Optional[bool] = None
    regularization: Optional[str] = None
    regularization_strength: Optional[float] = None
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
    lr: float = 0.001
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
    lr: float = 0.001
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
    eta_min: float = 0
    last_epoch: int = -1


@dataclass
class CosineAnnealingWarmRestartsConf:
    _target_: str = "schedulers.cosine_annealing_warm_restarts.CosineAnnealingWarmRestarts"
    T_0: int = MISSING
    T_mult: int = 1
    gamma: float = 1.0
    eta_min: float = 0
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
    gamma: float = 0.1
    last_epoch: int = -1


@dataclass
class MainSchedulerConf:
    name: MainSchedulerName = MISSING
    lr_min: Optional[float] = None
    restart_every: Optional[int] = None
    delay_restarts_by: Optional[int] = None
    lr_gamma: Optional[float] = None
    lr_step_size: Optional[float] = None
    kwargs: Union[
        CosineAnnealingLRConf,
        CosineAnnealingWarmRestartsConf,
        ExponentialLRConf,
        StepLRConf
    ] = MISSING


# -------------------------------
# WARMUP SCHEDULER CONFIGURATIONS
# -------------------------------

@dataclass
class ConstantLRConf:
    _target_: str = "torch.optim.lr_scheduler.ConstantLR"
    factor: float = 0.3333333333333333
    total_iters: int = 5
    last_epoch: int = -1


@dataclass
class LinearLRConf:
    _target_: str = "torch.optim.lr_scheduler.LinearLR"
    start_factor: float = 0.3333333333333333
    end_factor: float = 1.0
    total_iters: int = 5
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


# ---------------------
# OPTUNA CONFIGURATIONS
# ---------------------

@dataclass
class OptunaCategoricalConf:
    choices: Union[
        List[bool],
        List[int],
        List[float],
        List[str]
    ] = MISSING


@dataclass
class OptunaFloatConf:
    low: float = MISSING
    high: float = MISSING
    step: Optional[float] = None
    log: bool = False


@dataclass
class OptunaIntConf:
    low: int = MISSING
    high: int = MISSING
    step: int = 1
    log: bool = False


@dataclass
class OptunaParamConf:
    name: str = MISSING
    type: OptunaParamType = MISSING
    vals: Union[
        OptunaCategoricalConf,
        OptunaFloatConf,
        OptunaIntConf
    ] = MISSING


@dataclass
class RandomSamplerConf:
    _target_: str = "optuna.samplers._random.RandomSampler"
    seed: Optional[int] = None


@dataclass
class TPESamplerConf:
    _target_: str = "optuna.samplers._tpe.sampler.TPESampler"
    consider_prior: bool = True
    prior_weight: float = 1.0
    consider_magic_clip: bool = True
    consider_endpoints: bool = False
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    seed: Optional[int] = None
    multivariate: bool = False
    group: bool = False
    warn_independent_sampling: bool = True
    constant_liar: bool = False


@dataclass
class OptunaSamplerConf:
    name: OptunaSamplerName = MISSING
    n_startup_trials: Optional[int] = None
    multivariate: Optional[bool] = None
    kwargs: Union[
        RandomSamplerConf,
        TPESamplerConf
    ] = MISSING


@dataclass
class MedianPrunerConf:
    _target_: str = "optuna.pruners._median.MedianPruner"
    n_startup_trials: int = 5
    n_warmup_steps: int = 0
    interval_steps: int = 1
    n_min_trials: int = 1


@dataclass
class SuccessiveHalvingPrunerConf:
    _target_: str = "optuna.pruners._successive_halving.SuccessiveHalvingPruner"
    min_resource: Union[str, int] = "auto"
    reduction_factor: int = 4
    min_early_stopping_rate: int = 0
    bootstrap_count: int = 0


@dataclass
class OptunaPrunerConf:
    name: OptunaPrunerName = MISSING
    n_startup_trials: Optional[int] = None
    n_warmup_steps: Optional[int] = None
    interval_steps: Optional[int] = None
    n_min_trials: Optional[int] = None
    min_resource: Optional[Union[str, int]] = None
    reduction_factor: Optional[int] = None
    min_early_stopping_rate: Optional[int] = None
    kwargs: Union[
        MedianPrunerConf,
        SuccessiveHalvingPrunerConf
    ] = MISSING


@dataclass
class OptunaConf:
    study_name: str = MISSING
    minimize: bool = MISSING
    n_trials: int = MISSING
    params: List[OptunaParamConf] = MISSING
    sampler: OptunaSamplerConf = MISSING
    pruner: OptunaPrunerConf = MISSING


# ------------------------------
# REPRODUCIBILITY CONFIGURATIONS
# ------------------------------

@dataclass
class ReproducibilityConf:
    torch_seed: int = MISSING
    optuna_seed: int = MISSING
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


# --------------------------
# PERFORMANCE CONFIGURATIONS
# --------------------------

@dataclass
class Top1AccuracyConf:
    _target_: str = "torchmetrics.classification.MulticlassAccuracy"
    num_classes: int = MISSING
    top_k: int = 1
    average: str = "micro"
    multidim_average: str = "global"
    ignore_index: Optional[int] = None
    validate_args: bool = True


@dataclass
class Top5AccuracyConf:
    _target_: str = "torchmetrics.classification.MulticlassAccuracy"
    num_classes: int = MISSING
    top_k: int = 5
    average: str = "micro"
    multidim_average: str = "global"
    ignore_index: Optional[int] = None
    validate_args: bool = True


@dataclass
class PerformanceConf:
    metrics: Dict[str, Union[
        Top1AccuracyConf,
        Top5AccuracyConf
    ]] = MISSING
    evaluation_metric: str = MISSING
    higher_is_better: bool = MISSING
    min_delta: float = 0.0
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
    transform: TransformConf = MISSING
    dataloader: DataloaderConf = MISSING
    criterion: CriterionConf = MISSING
    performance: PerformanceConf = MISSING
    reproducibility: ReproducibilityConf = MISSING


@dataclass
class TrainClassifierConf(DictConfig):
    model: ModelConf = MISSING
    dataset: DatasetConf = MISSING
    transform: TransformConf = MISSING
    dataloader: DataloaderConf = MISSING
    criterion: CriterionConf = MISSING
    optimizer: OptimizerConf = MISSING
    main_scheduler: Optional[MainSchedulerConf] = None
    warmup_scheduler: Optional[WarmupSchedulerConf] = None
    experiment: ExperimentConf = MISSING
    optuna: Optional[OptunaConf] = None
    reproducibility: ReproducibilityConf = MISSING
    training: TrainingConf = MISSING
    performance: PerformanceConf = MISSING
    checkpoints: CheckpointsConf = MISSING
    tensorboard: TensorBoardConf = MISSING
    paths: PathsConf = MISSING


@dataclass
class TrainSimilarityConf(DictConfig):
    model: ModelConf = MISSING
    dataset: DatasetConf = MISSING
    transform: TransformConf = MISSING
    dataloader: DataloaderConf = MISSING
    repr_similarity: ReprSimilarityConf = MISSING
    optimizer: OptimizerConf = MISSING
    main_scheduler: Optional[MainSchedulerConf] = None
    warmup_scheduler: Optional[WarmupSchedulerConf] = None
    experiment: ExperimentConf = MISSING
    reproducibility: ReproducibilityConf = MISSING
    training: TrainingConf = MISSING
    performance: PerformanceConf = MISSING
    checkpoints: CheckpointsConf = MISSING
    tensorboard: TensorBoardConf = MISSING
    paths: PathsConf = MISSING
