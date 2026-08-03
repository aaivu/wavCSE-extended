"""
Evaluation utilities for multi task learning.

This module provides shared helper functions, data structures,
and writers used during downstream model evaluation, including:
- Batch and evaluation level statistics containers
- Safe numerical helpers and directory management
- Masked loss and accuracy computation for multi task settings
- Metric writing utilities
- CSV prediction writers for per task label and prediction export

These utilities are used by the downstream evaluator to ensure
consistent evaluation reporting and reproducible outputs.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn


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

    labels_task: Dict[int, torch.Tensor]
    preds_task: Dict[int, torch.Tensor]


@dataclass
class EvalStats:
    avg_loss_all: float
    accuracy_all: float
    avg_loss_task: Dict[int, float]
    accuracy_task: Dict[int, float]
    total_samples_task: Dict[int, int]


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_file_path(key: str, folder_path: str, file_format: str) -> str:
    ensure_dir(folder_path)
    return os.path.join(folder_path, f"{key}{file_format}")


def masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    ignore_index: int = -1
) -> Tuple[Optional[torch.Tensor], int]:
    """
    Returns:
      loss: Tensor or None if no valid labels
      valid_count: number of valid label positions
    """
    mask = labels != ignore_index
    labels_masked = labels[mask]
    valid = int(labels_masked.numel())

    if valid == 0:
        return None, 0

    # logits shape can be [B, C] or [B, L, C]
    # prediction is typically [B] or [B, L]
    logits_masked = logits[mask]
    loss = loss_fn(logits_masked, labels_masked)
    return loss, valid


def masked_accuracy(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1
) -> Tuple[int, int]:
    mask = labels != ignore_index
    labels_masked = labels[mask]
    pred_masked = prediction[mask]
    correct = int((pred_masked == labels_masked).sum().item())
    samples = int(labels_masked.numel())
    return correct, samples


class MetricsWriter:
    def __init__(self, metrics_path_all: str):
        self.metrics_path_all = metrics_path_all

    def write_metrics_all(self, line: str) -> None:
        with open(self.metrics_path_all, "a") as f:
            f.write(line + "\n")

            
class CsvPredictionWriter:
    """
    Writes labels and predictions to CSV files.
    One file per task.
    Naming:
      eval_predictions_{ckpt_tag}_{task}.csv
    """    
    def __init__(self, base_path: str, task_names: List[str], checkpoint_tag: str):
        self.base_path = base_path
        self.task_names = task_names
        self.checkpoint_tag = checkpoint_tag
    
    def _task_csv_path(self, task_name: str) -> str:
        return os.path.join(self.base_path, f"eval_predictions_{self.checkpoint_tag}_{task_name}.csv")

    def init_files(self) -> None:
        for tname in self.task_names:
            p = self._task_csv_path(tname)
            ensure_dir(os.path.dirname(p))
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([f"true_{tname}", f"pred_{tname}"])

    def append_row(self, task_name: str, true_val: Any, pred_val: Any) -> None:
        p = self._task_csv_path(task_name)
        with open(p, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([true_val, pred_val])

