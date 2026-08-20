"""
Training utilities for multi task learning.

This module provides shared helper functions, data structures,
and managers used during model training, including:
- Safe numerical helpers
- Task and epoch level statistics containers
- Masked loss and accuracy computation for multi task settings
- Metric file writers
- Checkpoint management for epoch, best, and optimal models

These utilities are used by the downstream training pipeline
to ensure consistent training behavior and evaluation tracking.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn


# ---------------------------
# helpers
# ---------------------------
def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def create_file_path(key: str, folder_path: str, file_format: str) -> str:
    os.makedirs(folder_path, exist_ok=True)
    file_name = f"train_{key}{file_format}"
    return os.path.join(folder_path, file_name)


def build_tasks_display_name(task_names):
    # "A", "A and B", "A, B and C", ...
    names = [str(x) for x in task_names if str(x).strip() != ""]
    if len(names) == 0:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


# ---------------------------
# dataclasses
# ---------------------------
@dataclass
class BatchStats:
    loss_all: float
    batch_all: int
    loss_task: Dict[int, float]
    batch_task: Dict[int, int]
    correct_task: Dict[int, int]
    samples_task: Dict[int, int]
    correct_all: int
    samples_all: int
    valid_count_task: Dict[int, int]


@dataclass
class EpochStats:
    avg_loss_all: float
    accuracy_all: float
    avg_loss_task: Dict[int, float]
    accuracy_task: Dict[int, float]
    total_samples_task: Dict[int, int]


# ---------------------------
# masked loss + accuracy
# ---------------------------
def masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    ignore_index: int = -1
) -> Tuple[Optional[torch.Tensor], int]:
    mask = labels != ignore_index
    labels_masked = labels[mask]
    if labels_masked.numel() == 0:
        return None, 0
    logits_masked = logits[mask, :]
    loss = loss_fn(logits_masked, labels_masked)
    return loss, int(labels_masked.size(0))


def masked_accuracy(
    pred: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1
) -> Tuple[int, int]:
    mask = labels != ignore_index
    labels_masked = labels[mask]
    if labels_masked.numel() == 0:
        return 0, 0
    pred_masked = pred[mask]
    correct = int((pred_masked == labels_masked).sum().item())
    total = int(labels_masked.size(0))
    return correct, total


# ---------------------------
# metric writer
# ---------------------------
class MetricsWriter:
    def __init__(self, metrics_path_all: str, metrics_path_task: Dict[int, str]):
        self.metrics_path_all = metrics_path_all
        self.metrics_path_task = metrics_path_task

        # ensure metric files exist (training only)
        for p in [metrics_path_all] + list(metrics_path_task.values()):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if not os.path.exists(p):
                with open(p, "w") as _:
                    pass

    def write_metrics_all(self, line: str) -> None:
        with open(self.metrics_path_all, "a") as f:
            print(line, file=f)

    def write_metrics_task(self, task_idx: int, line: str) -> None:
        path = self.metrics_path_task.get(task_idx)
        if path is None:
            return
        with open(path, "a") as f:
            print(line, file=f)


# ---------------------------
# checkpoint manager
# ---------------------------
class CheckpointManager:
    def __init__(self, model: nn.Module, model_checkpoint_path: str, saved_checkpoint_count: int):
        self.model = model
        self.model_checkpoint_path = model_checkpoint_path
        self.saved_checkpoint_count = saved_checkpoint_count

        self.best_accuracy_all_threshold = 0.0
        self.best_accuracy_task_thresholds: Dict[int, float] = {}

        self.opt_accuracy_all_threshold = 0.0
        self.opt_accuracy_task_thresholds: Dict[int, float] = {}
        self.opt_accuracy_avg_threshold = 0.0

    def save_epoch_checkpoint(self, epoch: int) -> None:
        path = self.model_checkpoint_path.replace(".pth", f"_epoch{epoch}.pth")
        torch.save(self.model.state_dict(), path)

        if epoch > self.saved_checkpoint_count:
            old_path = self.model_checkpoint_path.replace(".pth", f"_epoch{epoch - self.saved_checkpoint_count}.pth")
            if os.path.exists(old_path):
                os.remove(old_path)

    def maybe_save_best_and_opt(self, test_accuracy_all: float, test_accuracy_task: Dict[int, float], num_tasks: int) -> None:
        # BEST: based on "all"
        if test_accuracy_all > self.best_accuracy_all_threshold:
            self.best_accuracy_all_threshold = float(test_accuracy_all)
            for t in range(num_tasks):
                self.best_accuracy_task_thresholds[t] = float(test_accuracy_task.get(t, 0.0))
            best_path = self.model_checkpoint_path.replace(".pth", "_best.pth")
            torch.save(self.model.state_dict(), best_path)

        # OPT: based on average of tasks
        avg_acc = 0.0
        for t in range(num_tasks):
            avg_acc += float(test_accuracy_task.get(t, 0.0))
        avg_acc = avg_acc / float(num_tasks) if num_tasks > 0 else 0.0

        if avg_acc > self.opt_accuracy_avg_threshold:
            self.opt_accuracy_avg_threshold = float(avg_acc)
            self.opt_accuracy_all_threshold = float(test_accuracy_all)
            for t in range(num_tasks):
                self.opt_accuracy_task_thresholds[t] = float(test_accuracy_task.get(t, 0.0))
            opt_path = self.model_checkpoint_path.replace(".pth", "_opt.pth")
            torch.save(self.model.state_dict(), opt_path)
