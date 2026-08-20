"""
Shared MLflow logging helpers for wavCSE improvements runs.

Reusable by any owner folder under improvements/ (base, taskrelation,
clustering, lowrank, decomposition) without requiring changes to downstream/.
Every helper here only reads public attributes/return values already exposed
by MultiTasksModelTrainer and MultiTasksModelEvaluator (downstream/trainer,
downstream/evaluator) -- no subclassing.

Author: Kevin Sanjula
"""

import mlflow


def _flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def setup_mlflow(cfg):
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get("experiment_name", "default")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_config_params(cfg):
    flat = _flatten_dict({k: v for k, v in cfg.items() if k != "mlflow"})
    mlflow.log_params({k: v for k, v in flat.items() if v is not None})


def log_trainer_history(trainer):
    """Replays a FINISHED trainer's full per-epoch history into step-indexed MLflow
    metrics, all at once. Simple, but gives zero visibility if the process dies
    mid-training -- prefer log_epoch_stats() (below) for long/unattended runs."""
    task_array = trainer.task_array

    for epoch in range(trainer.num_epochs):
        metrics = {
            "train_loss_all": trainer.train_losses_all[epoch],
            "train_acc_all": trainer.train_acc_all[epoch],
            "val_loss_all": trainer.val_losses_all[epoch],
            "val_acc_all": trainer.val_acc_all[epoch],
            "learning_rate": trainer.learning_rate_array[epoch],
        }
        for t, task in enumerate(task_array):
            metrics[f"train_{task}_loss"] = trainer.train_losses_task[t][epoch]
            metrics[f"train_{task}_acc"] = trainer.train_acc_task[t][epoch]
            metrics[f"val_{task}_loss"] = trainer.val_losses_task[t][epoch]
            metrics[f"val_{task}_acc"] = trainer.val_acc_task[t][epoch]

        mlflow.log_metrics(metrics, step=epoch + 1)


def log_epoch_stats(epoch, phase, stats, task_array, learning_rate=None):
    """Logs one phase's ("train" or "val") EpochStats for one epoch immediately.

    Meant to be called live, during training, from a hook into one epoch at a
    time -- so a run killed mid-training still leaves a partial, inspectable
    metric curve on the tracking server, instead of nothing at all (see
    log_trainer_history's caveat above). See improvements/base/run_base.py's
    _LiveMlflowTrainer for the trainer-side hook this pairs with.
    """
    metrics = {
        f"{phase}_loss_all": stats.avg_loss_all,
        f"{phase}_acc_all": stats.accuracy_all,
    }
    for t, task in enumerate(task_array):
        metrics[f"{phase}_{task}_loss"] = stats.avg_loss_task[t]
        metrics[f"{phase}_{task}_acc"] = stats.accuracy_task[t]
    if learning_rate is not None:
        metrics["learning_rate"] = learning_rate

    mlflow.log_metrics(metrics, step=epoch)


def log_eval_stats(stats, tag, task_array):
    """Logs an EvalStats instance (from MultiTasksModelEvaluator.write_metrics()) under a checkpoint tag."""
    metrics = {
        f"test_{tag}_loss_all": stats.avg_loss_all,
        f"test_{tag}_acc_all": stats.accuracy_all,
    }
    for t, task in enumerate(task_array):
        metrics[f"test_{tag}_{task}_loss"] = stats.avg_loss_task[t]
        metrics[f"test_{tag}_{task}_acc"] = stats.accuracy_task[t]

    mlflow.log_metrics(metrics)
