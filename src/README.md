## Training Configuration ``cfg``


### ``BaseTrainer``

Passes *DictConfig* ``cfg`` to the [``CheckpointManager``](#checkpointmanager) class.

* checkpoints.save_best_model
* dataloader.batch_size
* paths.tensorboard
* performance.evaluate_on
* performance.evaluation_metric
* performance.higher_is_better
* performance.keep_previous_best_score
* performance.patience
* tensorboard.updates_per_epoch
* training.num_epochs
* training.resume_from


### ``CheckpointManager``

* checkpoints.delete_previous
* checkpoints.save_best_model
* checkpoints.save_frequency
* paths.checkpoints
* performance.evaluate_on
* performance.evaluation_metric


### ``ClassificationTrainer``

Passes *DictConfig* ``cfg`` to the [``BaseTrainer``](#basetrainer) class.


### ``RepresentationalSimilarityTrainer``

Passes *DictConfig* ``cfg`` to the [``BaseTrainer``](#basetrainer) class.

* repr_similarity.hooks.ref
* repr_similarity.hooks.train
