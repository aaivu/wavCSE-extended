"""Trainer that adds NCMTL clustering to the baseline training loop."""

import csv
import json
import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from trainer.trainer_model import MultiTasksModelTrainer
from trainer.trainer_utils import BatchStats, masked_accuracy, masked_ce_loss
from improvements.clustering.utils.ncmtl_clustering import cluster_candidate_weights


class MultiTasksModelTrainerNCMTL(MultiTasksModelTrainer):
    """Baseline trainer plus candidate-network clustering and projection."""

    def __init__(self, *args, ncmtl_cfg: dict, **kwargs):
        super().__init__(*args, **kwargs)
        if self.task_type != "ks_si_er":
            raise ValueError("NCMTL v1 supports only three tasks: task_type='ks_si_er'.")

        self.label_smoothing = float(self.training_cfg.get("label_smoothing", 0.0))
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must satisfy 0 <= value < 1")
        self.training_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing
        )
        # Keep validation loss unsmoothed so it remains comparable to test NLL
        # and continues to reveal confidence-related overfitting.
        self.validation_loss_fn = nn.CrossEntropyLoss()

        self.alpha = float(ncmtl_cfg.get("alpha", 0.001))
        self.num_clusters = int(ncmtl_cfg.get("num_clusters", 2))
        self.cluster_every_n_batches = int(
            ncmtl_cfg.get("cluster_every_n_batches", 1)
        )
        self.warmup_epochs = int(ncmtl_cfg.get("warmup_epochs", 0))
        self.kmeans_random_state = int(ncmtl_cfg.get("kmeans_random_state", 42))
        self.kmeans_n_init = int(ncmtl_cfg.get("kmeans_n_init", 1))
        self.kmeans_max_iter = int(ncmtl_cfg.get("kmeans_max_iter", 100))
        self.freeze_on_stability = bool(ncmtl_cfg.get("freeze_on_stability", True))
        self.stability_patience = int(ncmtl_cfg.get("stability_patience", 50))
        self.min_epochs_before_freeze = int(
            ncmtl_cfg.get("min_epochs_before_freeze", 1)
        )
        max_epochs = ncmtl_cfg.get("max_recluster_epochs", 4)
        self.max_recluster_epochs = None if max_epochs is None else int(max_epochs)

        if not 1 <= self.num_clusters <= 3:
            raise ValueError("num_clusters must satisfy 1 <= K <= 3")
        if self.cluster_every_n_batches <= 0:
            raise ValueError("cluster_every_n_batches must be positive")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative")
        if self.stability_patience <= 0:
            raise ValueError("stability_patience must be positive")
        if self.min_epochs_before_freeze < 0:
            raise ValueError("min_epochs_before_freeze cannot be negative")
        if self.max_recluster_epochs is not None and self.max_recluster_epochs <= 0:
            raise ValueError("max_recluster_epochs must be positive or null")

        self.current_epoch = 0
        self.current_batch = 0
        self._stable_update_count = 0
        self.frozen_epoch: Optional[int] = None
        self.cluster_history_path = os.path.join(self.results_dir, "cluster_history.csv")
        self.cluster_summary_path = os.path.join(self.results_dir, "cluster_summary.json")
        with open(self.cluster_history_path, "w", newline="") as history_file:
            csv.writer(history_file).writerow(
                ["epoch", "batch", "task", "cluster", "frozen", "cluster_loss"]
            )

        logging.info(
            "ncmtl_start | candidate_dim=%d | clusters=%d | alpha=%g | interval=%d | label_smoothing=%g",
            self.model.candidate_dim,
            self.num_clusters,
            self.alpha,
            self.cluster_every_n_batches,
            self.label_smoothing,
        )

    def _process_data_loader(self, data_loader, train_mode: bool):
        if train_mode:
            self.current_epoch += 1
            self.current_batch = 0
        stats = super()._process_data_loader(data_loader, train_mode=train_mode)
        # The epoch cap means reclustering remains active through the configured
        # epoch, then the learned assignment is fixed for following epochs.
        if (
            train_mode
            and self.max_recluster_epochs is not None
            and self.current_epoch >= self.max_recluster_epochs
            and self.model.has_valid_cluster_assignments()
            and not bool(self.model.cluster_frozen.item())
        ):
            self._freeze_clusters("maximum reclustering epoch reached")
        return stats

    def _freeze_clusters(self, reason: str) -> None:
        self.model.cluster_frozen.fill_(True)
        if self.frozen_epoch is None:
            self.frozen_epoch = self.current_epoch
        logging.info(
            "ncmtl_clusters_frozen | epoch=%d | batch=%d | reason=%s | assignments=%s",
            self.current_epoch,
            self.current_batch,
            reason,
            self.model.cluster_assignments.detach().cpu().tolist(),
        )

    def _should_recluster(self) -> bool:
        return (
            not bool(self.model.cluster_frozen.item())
            and self.current_epoch > self.warmup_epochs
            and self.current_batch % self.cluster_every_n_batches == 0
        )

    def _update_cluster_state(self) -> None:
        previous = None
        if self.model.has_valid_cluster_assignments():
            previous = self.model.cluster_assignments.detach().cpu().tolist()

        assignments = cluster_candidate_weights(
            self.model.get_flattened_candidate_weights(),
            num_clusters=self.num_clusters,
            random_state=self.kmeans_random_state,
            n_init=self.kmeans_n_init,
            max_iter=self.kmeans_max_iter,
            previous_assignments=previous,
        )

        if previous is not None and assignments == previous:
            self._stable_update_count += 1
        else:
            self._stable_update_count = 1

        self.model.set_cluster_assignments(assignments)
        self.model.share_candidate_weights_by_cluster()

        stable_enough = (
            self.freeze_on_stability
            and self.current_epoch >= self.min_epochs_before_freeze
            and self._stable_update_count >= self.stability_patience
        )
        if stable_enough:
            self._freeze_clusters("assignment stability patience reached")

    def _write_cluster_history(self, cluster_loss: float) -> None:
        if not self.model.has_valid_cluster_assignments():
            return
        assignments = self.model.cluster_assignments.detach().cpu().tolist()
        frozen = str(bool(self.model.cluster_frozen.item())).lower()
        with open(self.cluster_history_path, "a", newline="") as history_file:
            writer = csv.writer(history_file)
            for task, cluster in zip(self.task_array, assignments):
                writer.writerow(
                    [self.current_epoch, self.current_batch, task, cluster, frozen, cluster_loss]
                )

    def _process_batch(self, batch, train_mode: bool) -> BatchStats:
        input_seq, labels_list = self._unpack_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(input_seq=input_seq)
        logits_tuple = outputs.logits
        pred_tuple = outputs.prediction
        loss_weight = 1.0 / float(self.num_tasks)
        loss_all = torch.tensor(0.0, device=self.device)

        loss_task: Dict[int, float] = {}
        batch_task: Dict[int, int] = {}
        correct_task: Dict[int, int] = {}
        samples_task: Dict[int, int] = {}
        valid_count_task: Dict[int, int] = {}

        per_task_losses = {}
        prediction_loss_fn = (
            self.training_loss_fn if train_mode else self.validation_loss_fn
        )
        for task_index in range(self.num_tasks):
            loss, valid = masked_ce_loss(
                logits=logits_tuple[task_index],
                labels=labels_list[task_index],
                loss_fn=prediction_loss_fn,
                ignore_index=self.ignore_index,
            )
            per_task_losses[task_index] = loss
            valid_count_task[task_index] = valid

        for task_index in range(self.num_tasks):
            raw_loss = per_task_losses[task_index]
            task_loss, loss_all, present = self._calculate_total_loss(
                raw_loss, loss_all, loss_weight
            )
            loss_task[task_index] = float(task_loss.item()) if raw_loss is not None else 0.0
            batch_task[task_index] = int(present)

        cluster_loss = self.model.get_cluster_loss()
        loss_all = loss_all + self.alpha * cluster_loss

        l1_reg = torch.tensor(0.0, device=self.device)
        l2_reg = torch.tensor(0.0, device=self.device)
        for parameter in self.model.parameters():
            l1_reg = l1_reg + torch.sum(torch.abs(parameter))
            l2_reg = l2_reg + torch.sum(torch.square(parameter))
        loss_all = loss_all + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg

        if train_mode:
            self.current_batch += 1
            loss_all.backward()
            self.optimizer.step()
            if self._should_recluster():
                self._update_cluster_state()
                self._write_cluster_history(float(cluster_loss.detach().item()))
            elif bool(self.model.cluster_frozen.item()):
                self.model.share_candidate_weights_by_cluster()
                self._write_cluster_history(float(cluster_loss.detach().item()))

        correct_all = 0
        samples_all = 0
        for task_index in range(self.num_tasks):
            correct, samples = masked_accuracy(
                pred_tuple[task_index],
                labels_list[task_index],
                ignore_index=self.ignore_index,
            )
            correct_task[task_index] = correct
            samples_task[task_index] = samples
            correct_all += correct
            samples_all += samples

        self._save_debug_counts("train" if train_mode else "val", valid_count_task)
        return BatchStats(
            loss_all=float(loss_all.item()),
            batch_all=1,
            loss_task=loss_task,
            batch_task=batch_task,
            correct_task=correct_task,
            samples_task=samples_task,
            correct_all=correct_all,
            samples_all=samples_all,
            valid_count_task=valid_count_task,
        )

    def _epoch_report_line(self, epoch, phase, stats):
        try:
            import mlflow
            from improvements import mlflow_utils

            if mlflow.active_run() is not None:
                learning_rate = None
                if phase == "val":
                    learning_rate = float(self.optimizer.param_groups[0]["lr"])
                mlflow_utils.log_epoch_stats(
                    epoch, phase, stats, self.task_array, learning_rate=learning_rate
                )
        except Exception as error:
            logging.warning("NCMTL live MLflow metric logging failed: %s", error)
        return super()._epoch_report_line(epoch, phase, stats)

    def _write_cluster_summary(self) -> None:
        state = self.model.get_cluster_state()
        assignments = {
            task: cluster for task, cluster in zip(self.task_array, state["assignments"])
        }
        summary = {
            "task_type": self.task_type,
            "num_clusters": self.num_clusters,
            "frozen": state["frozen"],
            "frozen_epoch": self.frozen_epoch,
            "assignments": assignments,
            "kmeans_random_state": self.kmeans_random_state,
        }
        with open(self.cluster_summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2)

    def train(self) -> None:
        super().train()
        self._write_cluster_summary()
