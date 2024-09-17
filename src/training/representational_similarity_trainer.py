"""A trainer class for RSA-based training."""


from typing import Callable, Dict, Literal, Optional

import torch
from einops import rearrange
from omegaconf import DictConfig
from torch import nn
from torch.utils.hooks import RemovableHandle
from torchmetrics import MetricCollection
from typing_extensions import override

from src.base_classes.base_trainer import BaseTrainer
from src.training.helpers.metric_tracker import MetricTracker


class RepresentationalSimilarityTrainer(BaseTrainer):
    """A trainer class for RSA-based training.

    Attributes:
        activations: A dictionary storing intermediate activations that
          are used to compute RSA scores between the two models.
        checkpoint_manager: The CheckpointManager instance responsible
          for saving and loading model checkpoints.
        criterion: The criterion used for optimization.
        device: The device to train on.
        epoch_idx: The current epoch index, starting from 1.
        experiment_tracker: The ExperimentTracker instance to log
          results to TensorBoard.
        final_epoch_idx: The index of the final epoch.
        hooks: A dictionary storing the handles that can be used to
          remove the forward hooks attached to the two models (to
          extract intermediate activations).
        logger: The logger instance to record logs.
        metric_tracker: The MetricTracker instance to track performance
          metrics during training.
        model: The model to be trained.
        model_ref: The model that serves as a reference to compare
          against when assessing representational similarity.
        optimizer: The optimizer used during training.
        performance_tracker: The PerformanceTracker instance to monitor
          model performance and handle early stopping.
        preparation_time: A timestamp indicating the end of preparing a
          mini-batch (i.e., loading and moving to target device).
        processing_time: A timestamp indicating the end of processing a
          mini-batch.
        start_time: A timestamp indicating the start of processing a
          mini-batch.
        train_loader: The dataloader providing training samples.
        val_loader: The dataloader providing validation samples.

    Methods:
        eval_compute_efficiency(): Evaluate the compute efficiency for
          an individual batch.
        eval_epoch(): Evaluate the model on the validation set for a
          single epoch.
        get_pbar(dataloader, mode): Wrap the provided dataloader with a
          progress bar.
        record_timestamp(stage): Record timestamp to track compute
          efficiency.
        remove_hooks(): Remove all hooks.
        train(): Train the model for multiple epochs.
        train_epoch(): Train the model for a single epoch.
    """

    def __init__(
            self,
            model_train: nn.Module,
            model_ref: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: Callable,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            prediction_metrics: MetricCollection,
            device: torch.device,
            cfg: DictConfig,
            run_id: Optional[int] = None
    ) -> None:
        """Initialize the RepresentationalSimilarityTrainer instance.

        Note:
            The RepresentationalSimilarityTrainer instance passes the
            training configuration ``cfg`` to the BaseTrainer class
            during initialization.  Additionally, it makes direct use
            of the following entries of the configuration ``cfg``:
              * hooks.ref
              * hooks.train

        Args:
            model_train: The model to be trained.
            model_ref: The model that serves as a reference to compare
              against when assessing representational similarity.
            optimizer: The optimizer used during training.
            criterion: The criterion used for optimization.
            train_loader: The dataloader providing training samples.
            val_loader: The dataloader providing validation samples.
            prediction_metrics: The metrics to track during training
              that are computed from the model predictions and target
              values.
            device: The device to train on.
            cfg: The training configuration.
            run_id: Optional run ID to distinguish multiple runs using
              the same configuration.  Used to save checkpoints and
              event files in separate directories.
        """

        # MetricTracker
        self.metric_tracker = MetricTracker(
            mean_metrics=["Loss", "RSAScore"],
            prediction_metrics=prediction_metrics,
            device=device
        )
        self.metric_tracker.report_status()

        super().__init__(
            model=model_train,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            cfg=cfg,
            run_id=run_id
        )

        self.model_ref = model_ref
        self.model_ref.eval()
        self.model_ref.requires_grad_(False)

        # Attach hooks
        self.hooks = {}
        self.activations = {}
        self.hooks["Train"] = self._register_hook(
            model=self.model,
            activations=self.activations,
            layer=cfg.hooks.train,
            activations_key="Train"
        )
        self.hooks["Ref"] = self._register_hook(
            model=self.model_ref,
            activations=self.activations,
            layer=cfg.hooks.ref,
            activations_key="Ref"
        )

        self.criterion = criterion

    def remove_hooks(self) -> None:
        """Remove all hooks."""

        for handle in self.hooks.values():
            handle.remove()

    @override
    def _run_epoch(
            self,
            is_training: bool
    ) -> Dict[str, float]:
        mode: Literal["Train", "Val"] = "Train" if is_training else "Val"

        # Dataloader
        dataloader = self.train_loader if is_training else self.val_loader
        wrapped_loader = self.get_pbar(dataloader, mode)

        # Reset MetricTracker
        self.metric_tracker.reset(partial=True, total=True)

        # Set model being trained to appropriate mode
        self.model.train(is_training)

        # Loop over mini-batches
        with torch.set_grad_enabled(is_training):
            self.record_timestamp("start")
            for batch_idx, (inputs, targets) in enumerate(wrapped_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.record_timestamp("preparation")

                # Forward pass, extract intermediate activations
                predictions, _ = self.model(inputs), self.model_ref(inputs)
                activations, activations_ref = self.activations["Train"], self.activations["Ref"]

                # Reshape activations (rows = stimuli/images, columns = flattened activations)
                # NOTE: If activations are extracted from fully connected layers towards the end of
                #       networks, they already have the shape (batch_size, num_features) and no
                #       reshaping is needed.
                activations = self._reshape_activations(activations)
                activations_ref = self._reshape_activations(activations_ref)

                # Compute loss
                rsa_score, loss = self.criterion(
                    predictions=predictions,
                    targets=targets,
                    activations1=activations,
                    activations2=activations_ref
                )

                # Backward pass and optimization
                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.record_timestamp("processing")

                # Update MetricTracker
                self.metric_tracker.update(
                    mean_values={
                        "Loss": loss.item(),
                        "RSAScore": rsa_score.item()
                    },
                    predictions=predictions,
                    targets=targets
                )

                # Update progress bar
                mean_metrics_partial = self.metric_tracker.compute_mean_metrics("partial")
                prediction_metrics_partial = self.metric_tracker.compute_prediction_metrics(
                    "partial"
                )
                all_metrics = {
                    **mean_metrics_partial,
                    **prediction_metrics_partial,
                    "ComputeEfficiency": self.eval_compute_efficiency()
                }
                wrapped_loader.set_postfix(all_metrics)
                all_metrics.pop("ComputeEfficiency")

                # Log to TensorBoard
                if self.experiment_tracker.is_tracking:
                    # NOTE: The ``log_indices`` of the ExperimentTracker instance start from 1.
                    if batch_idx + 1 in self.experiment_tracker.log_indices[mode]:
                        # Log metrics to TensorBoard
                        self.experiment_tracker.log_scalars(
                            scalars=all_metrics,
                            step=self.get_global_step(
                                is_training=is_training,
                                batch_idx=batch_idx,
                                batch_size=len(targets)
                            ),
                            mode=mode
                        )

                        # Reset metrics for next set of mini-batches
                        self.metric_tracker.reset(partial=True)

                # Reset timer for next mini-batch
                self.record_timestamp("start")

            # Close progress bar and flush SummaryWriter
            wrapped_loader.close()
            self.experiment_tracker.flush()

            mean_metrics_total = self.metric_tracker.compute_mean_metrics("total")
            prediction_metrics_total = self.metric_tracker.compute_prediction_metrics("total")

            return {**mean_metrics_total, **prediction_metrics_total}

    @staticmethod
    def _register_hook(
            model: nn.Module,
            layer: str,
            activations: Dict[str, torch.Tensor],
            activations_key: str
    ) -> RemovableHandle:

        if layer not in [name for name, _ in model.named_modules()]:
            raise ValueError(
                f"{layer} is not the name of a layer of {model.__class__.__name__}."
            )

        def hook(module, args, output) -> None:
            activations[activations_key] = output

        module = model
        for name in layer.split("."):
            # NOTE: The following if-else block assumes that integers in the ``layer`` name are the
            #       indices of a Sequential module.
            if name.isdigit():
                module = module[int(name)]
            else:
                module = getattr(module, name)

        return module.register_forward_hook(hook)

    @staticmethod
    def _reshape_activations(activations: torch.Tensor) -> torch.Tensor:
        """Reshape activations if needed.

        Args:
            activations: The activations to reshape.  If the activations
              are extracted from convolutional layers, they are reshaped
              to have the shape (batch_size, num_features).
        """

        if activations.dim() == 4:
            return rearrange(activations, "b c h w -> b (c h w)")
        return activations
