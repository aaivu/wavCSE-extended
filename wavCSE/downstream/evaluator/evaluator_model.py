"""
Evaluator implementation for downstream multi task learning.

This module evaluates a trained downstream multi task model
using pre computed embeddings. It supports:
- Loading a specific checkpoint variant (opt, best, epoch, or epochX)
- Computing masked loss and accuracy for multi task settings
- Writing evaluation metrics to a text report
- Exporting per task labels and predictions to CSV files

The evaluator is designed to reuse the same run directory
created during training and to produce reproducible outputs.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn

from dataset.custom_emb_dataloader import CustomEmbDataLoader
from evaluator.evaluator_utils import (
    BatchStats,
    EvalStats,
    safe_div,
    create_file_path,
    masked_ce_loss,
    masked_accuracy,
    MetricsWriter,
    CsvPredictionWriter,
)

class MultiTasksModelEvaluator:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        task_type: str,
        evaluation_cfg: Dict[str, Any],     # loaded from evaluation.yml
        results_root: str,
        dataset,
        checkpoints_root: str,           
        checkpoint_tag: Optional[str] = None,
        ignore_index: int = -1,
        results_run_id: Optional[str] = None,
        checkpoint_run_id: Optional[str] = None,
    ):
        self.model = model
        self.device = device
        self.task_type = task_type
        
        self.task_array = [t for t in task_type.split("_") if t.strip() != ""]
        if len(self.task_array) == 0:
            raise ValueError(f"Invalid task_type: {task_type}")
        self.num_tasks = len(self.task_array)
        
        self.ignore_index = ignore_index

        batch_size = int(evaluation_cfg.get("batch_size", 1))
        shuffle = bool(evaluation_cfg.get("shuffle", False))
        pin_memory = bool(evaluation_cfg.get("pin_memory", True))
        drop_last = bool(evaluation_cfg.get("drop_last", False))
        num_workers = int(evaluation_cfg.get("num_workers", 0))

        # self.save_metrics = bool(evaluation_cfg.get("save_metrics", True))
        # self.save_predictions = bool(evaluation_cfg.get("save_predictions", True))

        # ----------------------------
        # use existing run dirs (no new run_id)
        # ----------------------------
        if results_run_id is None:
            results_run_id = self._latest_run_id(results_root)

        if checkpoint_run_id is None:
            checkpoint_run_id = self._latest_run_id(checkpoints_root)

        self.results_dir = os.path.join(results_root, results_run_id)
        self.checkpoints_dir = os.path.join(checkpoints_root, checkpoint_run_id)

        if not os.path.isdir(self.results_dir):
            raise FileNotFoundError(f"results_dir not found: {self.results_dir}")
        if not os.path.isdir(self.checkpoints_dir):
            raise FileNotFoundError(f"checkpoints_dir not found: {self.checkpoints_dir}")

        # ----------------------------
        # paths: metrics txt
        # ----------------------------
        self.metrics_path_all = create_file_path(
            key=f"eval_metrics_{checkpoint_tag}",
            folder_path=self.results_dir,
            file_format=".txt"
        )

        self.metrics_writer = MetricsWriter(
            metrics_path_all=self.metrics_path_all
        )

        # ----------------------------
        # paths: prediction csv
        # ----------------------------
        self.csv_writer = CsvPredictionWriter(
            base_path=self.results_dir,
            task_names=self.task_array,
            checkpoint_tag=checkpoint_tag
        )

        # ----------------------------
        # dataloader
        # Expected batch format: (x, labels_tensor[B, T])
        # This matches your trainer style.
        # ----------------------------
        self.dataloader = CustomEmbDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        # ----------------------------
        # load checkpoint
        # ----------------------------
        if checkpoints_root is None:
            raise ValueError("checkpoints_root cannot be None")
        if checkpoint_tag is None:
            checkpoint_tag = "opt"

        self.checkpoint_tag = checkpoint_tag

        ckpt_path = self._resolve_checkpoint_path(self.checkpoints_dir, checkpoint_tag)

        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        self.loss_fn = nn.CrossEntropyLoss()

        # log start
        logging.info(
            "eval_start | results_run_id=%s | checkpoint_run_id=%s | task_type=%s | tasks=%s | checkpoint_tag=%s | ckpt=%s | results_dir=%s",
            results_run_id, checkpoint_run_id, self.task_type, self.task_array, checkpoint_tag, ckpt_path, self.results_dir
        )
        
    def _latest_run_id(self, root_dir: str) -> str:
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root dir not found: {root_dir}")

        run_dirs = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        if len(run_dirs) == 0:
            raise FileNotFoundError(f"No run directories found in: {root_dir}")

        run_dirs = sorted(
            run_dirs,
            key=lambda d: os.path.getmtime(os.path.join(root_dir, d)),
            reverse=True
        )
        return run_dirs[0]
        
    def _resolve_checkpoint_path(self, checkpoints_root: str, checkpoint_tag: str) -> str:
        if not os.path.isdir(checkpoints_root):
            raise FileNotFoundError(f"checkpoints_root not found: {checkpoints_root}")

        tag = checkpoint_tag.lower().strip()

        ckpt_files = [f for f in os.listdir(checkpoints_root) if f.endswith(".pth")]

        matches = [f for f in ckpt_files if tag in f.lower()]

        if len(matches) == 0:
            raise FileNotFoundError(
                f"No checkpoint containing '{checkpoint_tag}' found in {checkpoints_root}. "
                f"Available: {ckpt_files}"
            )

        # If multiple match (rare), take newest
        matches = sorted(
            matches,
            key=lambda f: os.path.getmtime(os.path.join(checkpoints_root, f)),
            reverse=True
        )

        return os.path.join(checkpoints_root, matches[0])

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x, labels_tensor = batch

        x = x.to(self.device)

        if not isinstance(labels_tensor, torch.Tensor) or labels_tensor.dim() != 2:
            raise ValueError("Second element must be a tensor of shape [B, T]")

        labels_tensor = labels_tensor.to(self.device)
        labels_list = [labels_tensor[:, t] for t in range(self.num_tasks)]
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

    def _process_batch(self, batch) -> BatchStats:
        x, labels_list = self._unpack_batch(batch)

        with torch.no_grad():
            outputs = self.model(input_seq=x)
            logits_tuple = outputs.logits
            pred_tuple = outputs.prediction

            loss_all = torch.tensor(0.0, device=self.device)
            batch_all = 1
            loss_weight = 1.0 / float(self.num_tasks)

            loss_task: Dict[int, float] = {}
            batch_task: Dict[int, int] = {}
            correct_task: Dict[int, int] = {}
            samples_task: Dict[int, int] = {}

            labels_out: Dict[int, torch.Tensor] = {}
            preds_out: Dict[int, torch.Tensor] = {}

            per_task_losses: Dict[int, Optional[torch.Tensor]] = {}
            for t in range(self.num_tasks):
                lt, _valid = masked_ce_loss(
                    logits=logits_tuple[t],
                    labels=labels_list[t],
                    loss_fn=self.loss_fn,
                    ignore_index=self.ignore_index
                )
                per_task_losses[t] = lt

            for t in range(self.num_tasks):
                lt, loss_all, bt = self._calculate_total_loss(per_task_losses[t], loss_all, loss_weight)
                loss_task[t] = float(lt.item())
                batch_task[t] = int(bt)

            correct_all = 0
            samples_all = 0
            for t in range(self.num_tasks):
                c, s = masked_accuracy(pred_tuple[t], labels_list[t], ignore_index=self.ignore_index)
                correct_task[t] = c
                samples_task[t] = s
                correct_all += c
                samples_all += s

                # store masked labels and preds for CSV writing
                mask = labels_list[t] != self.ignore_index
                labels_out[t] = labels_list[t][mask].detach().cpu()
                preds_out[t] = pred_tuple[t][mask].detach().cpu()

        return BatchStats(
            loss_all=float(loss_all.item()),
            batch_all=batch_all,
            loss_task=loss_task,
            batch_task=batch_task,
            correct_task=correct_task,
            samples_task=samples_task,
            correct_all=correct_all,
            samples_all=samples_all,
            labels_task=labels_out,
            preds_task=preds_out,
        )

    def _process_data_loader(self) -> EvalStats:
        total_loss_all = 0.0
        total_batches_all = 0
        total_correct_all = 0
        total_samples_all = 0

        total_loss_task: Dict[int, float] = {t: 0.0 for t in range(self.num_tasks)}
        total_batches_task: Dict[int, int] = {t: 0 for t in range(self.num_tasks)}
        total_correct_task: Dict[int, int] = {t: 0 for t in range(self.num_tasks)}
        total_samples_task: Dict[int, int] = {t: 0 for t in range(self.num_tasks)}

        for batch in self.dataloader:
            bs = self._process_batch(batch)

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

        return EvalStats(
            avg_loss_all=float(avg_loss_all),
            accuracy_all=float(acc_all),
            avg_loss_task={t: float(avg_loss_task[t]) for t in range(self.num_tasks)},
            accuracy_task={t: float(acc_task[t]) for t in range(self.num_tasks)},
            total_samples_task=total_samples_task
        )

    def write_metrics(self) -> EvalStats:
        stats = self._process_data_loader()

        # all file: all metrics only
        lines = []
        lines.append(f"loss_all={stats.avg_loss_all:.6f} | acc_all={stats.accuracy_all:.6f}")

        for t in range(self.num_tasks):
            tname = self.task_array[t]
            lines.append(
                f"{tname} | loss={stats.avg_loss_task[t]:.6f} | acc={stats.accuracy_task[t]:.6f} | samples={stats.total_samples_task[t]}"
            )

        for line in lines:
            self.metrics_writer.write_metrics_all(line)

        # overall metrics
        logging.info(
            "eval_metrics_written | loss_all=%.6f acc_all=%.6f",
            stats.avg_loss_all,
            stats.accuracy_all
        )

        # per-task metrics
        for t, task in enumerate(self.task_array):
            logging.info(
                "eval_task | task=%s | loss=%.6f | acc=%.6f | samples=%d",
                task,
                stats.avg_loss_task[t],
                stats.accuracy_task[t],
                stats.total_samples_task[t]
            )

        return stats

    def write_predictions_csv(self) -> None:
        self.csv_writer.init_files()

        for batch in self.dataloader:
            bs = self._process_batch(batch)

            for t in range(self.num_tasks):
                tname = self.task_array[t]
                labels = bs.labels_task[t]
                preds = bs.preds_task[t]

                # write one row per valid label
                for i in range(labels.numel()):
                    self.csv_writer.append_row(tname, int(labels[i].item()), int(preds[i].item()))

        logging.info("eval_predictions_saved")

    def run(self, save_metrics: bool = True, save_predictions: bool = True) -> None:
        if save_metrics:
            self.write_metrics()
        if save_predictions:
            self.write_predictions_csv()

        logging.info("eval_end")

