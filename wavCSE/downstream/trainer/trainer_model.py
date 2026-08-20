"""
Trainer implementation for downstream multi task learning.

This module implements the full training loop for a downstream
multi task classification model, including:
- DataLoader construction
- Training and validation epochs
- Masked loss and accuracy computation
- Learning rate scheduling
- Checkpoint management (epoch, best, optimal)
- Metric logging and visualization

The trainer operates on pre computed embeddings and supports
an arbitrary number of tasks.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

from trainer.trainer_utils import (
    safe_div,
    create_file_path,
    build_tasks_display_name,
    BatchStats,
    EpochStats,
    masked_ce_loss,
    masked_accuracy,
    MetricsWriter,
    CheckpointManager,
)

from utils.constant_mapping import TaskKeywordMapping
from dataset.custom_emb_dataloader import CustomEmbDataLoader

class MultiTasksModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        task_type: str,
        training_cfg: Dict[str, Any],     
        results_root: str,
        checkpoints_root: str,
        training_data=None,
        validation_data=None,
        ignore_index: int = -1,
    ):
        self.model = model
        self.device = device
        self.ignore_index = ignore_index

        # ----------------------------
        # tasks
        # ----------------------------
        self.task_type = task_type
        self.task_array = [t for t in task_type.split("_") if t.strip() != ""]
        if len(self.task_array) == 0:
            raise ValueError(f"Invalid task_type: {task_type}")
        self.num_tasks = len(self.task_array)

        # friendly task names for plot titles
        self.task_name_array = [
            TaskKeywordMapping.get_task_name(t) or t for t in self.task_array
        ]
        self.tasks_display_name = build_tasks_display_name(self.task_name_array)

        # ----------------------------
        # run dirs
        # ----------------------------
        run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.results_dir = os.path.join(results_root, run_id)
        self.ckpt_dir = os.path.join(checkpoints_root, run_id)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # ----------------------------
        # training cfg (supports flat OR nested)
        # ----------------------------
        self.training_cfg = training_cfg

        self.num_epochs = int(training_cfg["num_epochs"])
        self.saved_checkpoint_count = int(training_cfg.get("saved_checkpoint_count", 5))

        # batch / loader params
        self.batch_size = int(training_cfg["batch_size"])
        self.shuffle_train = bool(training_cfg.get("shuffle_train", True))
        self.shuffle_val = bool(training_cfg.get("shuffle_val", False))  # you can override
        self.pin_memory = bool(training_cfg.get("pin_memory", True))
        self.drop_last_train = bool(training_cfg.get("drop_last_train", True))
        self.drop_last_val = bool(training_cfg.get("drop_last_val", False))
        self.num_workers = int(training_cfg.get("num_workers", 0))

        # regularization (flat only)
        self.l1_lambda = float(training_cfg.get("l1_lambda", 0.0))
        self.l2_lambda = float(training_cfg.get("l2_lambda", 0.0))

        # optimizer (flat only)
        lr = float(training_cfg["learning_rate"])
        wd = float(training_cfg.get("weight_decay", 0.0))

        # scheduler (flat only)
        patience = int(training_cfg.get("scheduler_patience", 1))
        factor = float(training_cfg.get("scheduler_factor", 0.5))

        # ----------------------------
        # create all file paths ONCE
        # (no upstream_model_type)
        # ----------------------------
        # all-tasks report and plot
        self.result_text_path_all = create_file_path(
            key=self.task_type,
            folder_path=self.results_dir,
            file_format=".txt",
        )
        self.result_plot_path_all = create_file_path(
            key=self.task_type,
            folder_path=self.results_dir,
            file_format=".png",
        )

        # per-task report and plot (works for any number of tasks)
        self.result_text_path_task: Dict[int, str] = {}
        self.result_plot_path_task: Dict[int, str] = {}
        for i, t in enumerate(self.task_array):
            self.result_text_path_task[i] = create_file_path(t, self.results_dir, ".txt")
            self.result_plot_path_task[i] = create_file_path(t, self.results_dir, ".png")

        # checkpoint path (base)
        self.model_checkpoint_path = create_file_path(
            key=self.task_type,
            folder_path=self.ckpt_dir,
            file_format=".pth",
        )

        # data_count path
        self.data_count_path = create_file_path(
            key="data_count",
            folder_path=self.results_dir,
            file_format=".txt",
        )

        # ----------------------------
        # optimizer + scheduler
        # ----------------------------
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.loss_fn = nn.CrossEntropyLoss()

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=patience,
            factor=factor
        )

        # ----------------------------
        # checkpoints
        # ----------------------------
        self.ckpt = CheckpointManager(
            model=self.model,
            model_checkpoint_path=self.model_checkpoint_path,
            saved_checkpoint_count=self.saved_checkpoint_count
        )

        # ----------------------------
        # history for plots
        # ----------------------------
        self.train_losses_all: List[float] = []
        self.train_acc_all: List[float] = []
        self.val_losses_all: List[float] = []
        self.val_acc_all: List[float] = []

        self.train_losses_task: Dict[int, List[float]] = {t: [] for t in range(self.num_tasks)}
        self.train_acc_task: Dict[int, List[float]] = {t: [] for t in range(self.num_tasks)}
        self.val_losses_task: Dict[int, List[float]] = {t: [] for t in range(self.num_tasks)}
        self.val_acc_task: Dict[int, List[float]] = {t: [] for t in range(self.num_tasks)}

        self.learning_rate_array: List[float] = []

        # ----------------------------
        # dataloaders are created HERE
        # ----------------------------
        if training_data is None or validation_data is None:
            raise ValueError("Pass training_data and validation_data (not dataloaders).")

        self.train_dataloader = CustomEmbDataLoader(
            training_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last_train,
            num_workers=self.num_workers,
        )

        self.val_dataloader = CustomEmbDataLoader(
            validation_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,   # you asked shuffle=True, but val shuffle normally False
            pin_memory=self.pin_memory,
            drop_last=self.drop_last_val,
            num_workers=self.num_workers,
        )
        
        # ----------------------------
        # metrics writer (training only)
        # ----------------------------
        self.metrics_writer = MetricsWriter(
            metrics_path_all=self.result_text_path_all,
            metrics_path_task=self.result_text_path_task
        )

        # ----------------------------
        # log run start
        # ----------------------------
        logging.info(model)
        logging.info(f"l1 lambda: {self.l1_lambda}")
        logging.info(f"l2 lambda: {self.l2_lambda}")
        logging.info(f"Result directory: {self.results_dir}")
        logging.info(f"Checkpoint directory: {self.ckpt_dir}")

    # ----------------------------
    # internals
    # ----------------------------
    def _save_debug_counts(self, operation: str, valid_counts: Dict[int, int]) -> None:
        parts = [operation] + [str(valid_counts.get(t, 0)) for t in range(self.num_tasks)]
        with open(self.data_count_path, "a") as f:
            print(" ".join(parts), file=f)
    
    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = batch[0].to(self.device)

        # ONLY support: (x, labels_tensor[B, T])
        if len(batch) != 2:
            raise ValueError("Batch must be (input_seq, labels_tensor[B,T])")

        labels_tensor = batch[1]
        if not isinstance(labels_tensor, torch.Tensor) or labels_tensor.dim() != 2:
            raise ValueError("Second element must be a tensor of shape [B, T]")

        labels_tensor = labels_tensor.to(self.device)
        labels_list = [labels_tensor[:, t] for t in range(labels_tensor.size(1))]

        return x, labels_list

    def _calculate_total_loss(
        self,
        loss_task: Optional[torch.Tensor],
        loss_all: torch.Tensor,
        loss_weight: float
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        batch_task = int(loss_task is not None)
        if loss_task is None:
            loss_task = torch.tensor(0.0, device=self.device)
        else:
            loss_all = loss_all + loss_task * loss_weight
        return loss_task, loss_all, batch_task

    def _process_batch(self, batch, train_mode: bool) -> BatchStats:
        input_seq, labels_list = self._unpack_batch(batch)

        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(input_seq=input_seq)
        logits_tuple = outputs.logits
        pred_tuple = outputs.prediction

        loss_weight = 1.0 / float(self.num_tasks)

        loss_all = torch.tensor(0.0, device=self.device)
        batch_all = 1

        loss_task: Dict[int, float] = {}
        batch_task: Dict[int, int] = {}
        correct_task: Dict[int, int] = {}
        samples_task: Dict[int, int] = {}
        valid_count_task: Dict[int, int] = {}

        per_task_losses: Dict[int, Optional[torch.Tensor]] = {}
        for t in range(self.num_tasks):
            loss_t, valid = masked_ce_loss(
                logits=logits_tuple[t],
                labels=labels_list[t],
                loss_fn=self.loss_fn,
                ignore_index=self.ignore_index
            )
            per_task_losses[t] = loss_t
            valid_count_task[t] = valid
            
        for t in range(self.num_tasks):
            lt_raw = per_task_losses[t]
            lt, loss_all, bt = self._calculate_total_loss(lt_raw, loss_all, loss_weight)
            loss_task[t] = float(lt.item()) if lt_raw is not None else 0.0
            batch_task[t] = int(bt)

        # L1 + L2
        l1_reg = torch.tensor(0.0, device=self.device)
        l2_reg = torch.tensor(0.0, device=self.device)
        for p in self.model.parameters():
            l1_reg = l1_reg + torch.sum(torch.abs(p))
            l2_reg = l2_reg + torch.sum(torch.square(p))
        loss_all = loss_all + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg

        if train_mode:
            loss_all.backward()
            self.optimizer.step()

        correct_all = 0
        samples_all = 0
        for t in range(self.num_tasks):
            c, s = masked_accuracy(pred_tuple[t], labels_list[t], ignore_index=self.ignore_index)
            correct_task[t] = c
            samples_task[t] = s
            correct_all += c
            samples_all += s

        self._save_debug_counts("train" if train_mode else "val", valid_count_task)

        return BatchStats(
            loss_all=float(loss_all.item()),
            batch_all=batch_all,
            loss_task=loss_task,
            batch_task=batch_task,
            correct_task=correct_task,
            samples_task=samples_task,
            correct_all=correct_all,
            samples_all=samples_all,
            valid_count_task=valid_count_task
        )

    def _process_data_loader(self, data_loader, train_mode: bool) -> EpochStats:
        self.model.train() if train_mode else self.model.eval()

        total_loss_all = 0.0
        total_batches_all = 0
        total_correct_all = 0
        total_samples_all = 0

        total_loss_task: Dict[int, float] = {t: 0.0 for t in range(self.num_tasks)}
        total_batches_task: Dict[int, int] = {t: 0 for t in range(self.num_tasks)}
        total_correct_task: Dict[int, int] = {t: 0 for t in range(self.num_tasks)}
        total_samples_task: Dict[int, int] = {t: 0 for t in range(self.num_tasks)}

        with torch.set_grad_enabled(train_mode):
            for batch in data_loader:
                bs = self._process_batch(batch, train_mode=train_mode)

                total_loss_all += bs.loss_all
                total_batches_all += bs.batch_all
                total_correct_all += bs.correct_all
                total_samples_all += bs.samples_all

                for t in range(self.num_tasks):
                    total_loss_task[t] += bs.loss_task[t]
                    total_batches_task[t] += bs.batch_task[t]
                    total_correct_task[t] += bs.correct_task[t]
                    total_samples_task[t] += bs.samples_task[t]

        acc_all = safe_div(total_correct_all, total_samples_all)
        avg_loss_all = safe_div(total_loss_all, total_batches_all)

        acc_task = {t: safe_div(total_correct_task[t], total_samples_task[t]) for t in range(self.num_tasks)}
        avg_loss_task = {t: safe_div(total_loss_task[t], total_batches_task[t]) for t in range(self.num_tasks)}

        return EpochStats(
            avg_loss_all=float(avg_loss_all),
            accuracy_all=float(acc_all),
            avg_loss_task={t: float(avg_loss_task[t]) for t in range(self.num_tasks)},
            accuracy_task={t: float(acc_task[t]) for t in range(self.num_tasks)},
            total_samples_task=total_samples_task
        )

    def _epoch_report_line(self, epoch: int, phase: str, stats: EpochStats) -> str:
        parts = [
            f"Epoch {epoch}/{self.num_epochs}",
            phase,
            f"loss_all={stats.avg_loss_all:.4f}",
            f"acc_all={stats.accuracy_all:.4f}",
        ]
        for t, task in enumerate(self.task_array):
            parts.append(f"{task}_loss={stats.avg_loss_task[t]:.4f}")
            parts.append(f"{task}_acc={stats.accuracy_task[t]:.4f}")
        return " | ".join(parts)

    def train(self) -> None:
        logging.info(
            "training_start | task_type=%s | num_tasks=%d | epochs=%d | batch_size=%d | device=%s",
            self.task_type,
            self.num_tasks,
            self.num_epochs,
            self.batch_size,
            str(self.device)
        )
        
        for ep in range(1, self.num_epochs + 1):
            logging.info("epoch_start | epoch=%d/%d", ep, self.num_epochs)
            train_stats = self._process_data_loader(self.train_dataloader, train_mode=True)
            val_stats = self._process_data_loader(self.val_dataloader, train_mode=False)

            # history
            self.train_losses_all.append(train_stats.avg_loss_all)
            self.train_acc_all.append(train_stats.accuracy_all)
            self.val_losses_all.append(val_stats.avg_loss_all)
            self.val_acc_all.append(val_stats.accuracy_all)

            for t in range(self.num_tasks):
                self.train_losses_task[t].append(train_stats.avg_loss_task[t])
                self.train_acc_task[t].append(train_stats.accuracy_task[t])
                self.val_losses_task[t].append(val_stats.avg_loss_task[t])
                self.val_acc_task[t].append(val_stats.accuracy_task[t])

            # metrics -> all.txt + each task txt
            self.metrics_writer.write_metrics_all(self._epoch_report_line(ep, "train", train_stats))
            self.metrics_writer.write_metrics_all(self._epoch_report_line(ep, "val", val_stats))

            for t, task in enumerate(self.task_array):
                # per-task short line
                self.metrics_writer.write_metrics_task(
                    t,
                    f"Epoch {ep}/{self.num_epochs} | "
                    f"{task} | train_loss={train_stats.avg_loss_task[t]:.4f} train_acc={train_stats.accuracy_task[t]:.4f} | "
                    f"val_loss={val_stats.avg_loss_task[t]:.4f} val_acc={val_stats.accuracy_task[t]:.4f}"
                )

            # log -> run.log
            lr = float(self.optimizer.param_groups[0]["lr"])
            self.learning_rate_array.append(lr)
            
            logging.info(
                "epoch_all | epoch=%d | lr=%.6g | train(loss=%.4f acc=%.4f) | val(loss=%.4f acc=%.4f)",
                ep, lr,
                train_stats.avg_loss_all, train_stats.accuracy_all,
                val_stats.avg_loss_all, val_stats.accuracy_all
            )

            for t, task in enumerate(self.task_array):
                logging.info(
                    "epoch_task | epoch=%d | task=%s | train(loss=%.4f acc=%.4f) | val(loss=%.4f acc=%.4f) | samples=%d",
                    ep, task,
                    train_stats.avg_loss_task[t], train_stats.accuracy_task[t],
                    val_stats.avg_loss_task[t], val_stats.accuracy_task[t],
                    int(val_stats.total_samples_task.get(t, 0))
                )

            # checkpoints + scheduler
            self.ckpt.save_epoch_checkpoint(ep)
            self.ckpt.maybe_save_best_and_opt(
                test_accuracy_all=val_stats.accuracy_all,
                test_accuracy_task=val_stats.accuracy_task,
                num_tasks=self.num_tasks
            )
            self.scheduler.step(val_stats.avg_loss_all)

        # final summary -> txt
        self.metrics_writer.write_metrics_all(f"Best model accuracy - all: {self.ckpt.best_accuracy_all_threshold:.6f}")
        for t, task in enumerate(self.task_array):
            self.metrics_writer.write_metrics_all(
                f"Best model accuracy - {task}: {self.ckpt.best_accuracy_task_thresholds.get(t, 0.0):.6f}"
            )

        self.metrics_writer.write_metrics_all(f"Opt model accuracy - all: {self.ckpt.opt_accuracy_all_threshold:.6f}")
        for t, task in enumerate(self.task_array):
            self.metrics_writer.write_metrics_all(
                f"Opt model accuracy - {task}: {self.ckpt.opt_accuracy_task_thresholds.get(t, 0.0):.6f}"
            )

        logging.info("training_end | run_id=%s | best_all=%.6f | opt_all=%.6f",
             os.path.basename(self.results_dir),
             self.ckpt.best_accuracy_all_threshold,
             self.ckpt.opt_accuracy_all_threshold)
        
        self.plot_metrics()

    # ----------------------------
    # plotting
    # ----------------------------
    def _plot_evalmetric(self, num_epochs, train_data, val_data, plot_path, plot_title, metric_name):
        plt.figure(figsize=(16, 8))
        plt.plot(range(1, num_epochs + 1), train_data, label=f"Train {metric_name}", marker="o")
        plt.plot(range(1, num_epochs + 1), val_data, label=f"Val {metric_name}", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        plt.xticks(range(1, num_epochs + 1))
        plt.title(plot_title)
        plt.tight_layout()
        out_path = plot_path.replace(".png", f"_{metric_name}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, transparent=True)
        plt.close()

    def _plot_lr(self, num_epochs, lr_array, plot_path, plot_title):
        plt.figure(figsize=(16, 8))
        plt.plot(range(1, num_epochs + 1), lr_array, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.xticks(range(1, num_epochs + 1))
        plt.yscale("log")
        plt.gca().yaxis.set_major_locator(LogLocator())
        plt.title(plot_title)
        plt.tight_layout()
        out_path = plot_path.replace(".png", "_lr.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, transparent=True)
        plt.close()

    def plot_metrics(self) -> None:
        # title for "all"
        all_title = f"{self.tasks_display_name}"

        self._plot_evalmetric(
            self.num_epochs,
            self.train_acc_all,
            self.val_acc_all,
            self.result_plot_path_all,
            all_title,
            "Accuracy"
        )
        self._plot_evalmetric(
            self.num_epochs,
            self.train_losses_all,
            self.val_losses_all,
            self.result_plot_path_all,
            all_title,
            "Loss"
        )
        self._plot_lr(
            self.num_epochs,
            self.learning_rate_array,
            self.result_plot_path_all,
            all_title
        )

        # per task
        for t, task in enumerate(self.task_array):
            title = TaskKeywordMapping.get_task_name(task) or task
            self._plot_evalmetric(
                self.num_epochs,
                self.train_acc_task[t],
                self.val_acc_task[t],
                self.result_plot_path_task[t],
                title,
                "Accuracy"
            )
            self._plot_evalmetric(
                self.num_epochs,
                self.train_losses_task[t],
                self.val_losses_task[t],
                self.result_plot_path_task[t],
                title,
                "Loss"
            )
