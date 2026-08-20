"""
PMR Trainer: Alternating minimization for wavCSE-PMR.

Inspired by Goncalves et al. (2016) Algorithm in Section 4.1:
Alternating between:
1. Update W (task parameters) with Omega fixed -> standard training step
2. Update Omega (precision matrix) with W fixed -> ADMM-style update

The precision matrix loss is added to the standard classification loss:
  L_total = L_classification + gamma * (-log|Omega| + Tr(W Omega W^T)) + lambda * |Omega|_1
"""

import os
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../downstream'))

from trainer.trainer_model import MultiTasksModelTrainer


class MultiTasksModelTrainerPMR(MultiTasksModelTrainer):
    """
    Trainer for wavCSE-PMR with precision matrix regularization.

    The precision matrix Omega is learned jointly with model parameters.
    At each batch, we add the PMR loss to the standard classification loss.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        task_type: str,
        training_cfg: Dict,
        results_root: str,
        checkpoints_root: str,
        training_data=None,
        validation_data=None,
        ignore_index: int = -1,
        pmr_warmup_epochs: int = 3,  # Number of epochs before enabling PMR loss
    ):
        super().__init__(
            model=model,
            device=device,
            task_type=task_type,
            training_cfg=training_cfg,
            results_root=results_root,
            checkpoints_root=checkpoints_root,
            training_data=training_data,
            validation_data=validation_data,
            ignore_index=ignore_index,
        )

        self.pmr_warmup_epochs = pmr_warmup_epochs
        self.current_epoch = 0

        logging.info(f"PMR warmup epochs: {pmr_warmup_epochs}")

    def _process_batch(self, batch, train_mode: bool):
        """
        Override batch processing to add PMR loss term.

        The PMR loss encourages task parameters to have a sparse precision
        structure, where Omega_ij = 0 means tasks i, j are conditionally
        independent.
        """
        input_seq, labels_list = self._unpack_batch(batch)

        if train_mode:
            self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(input_seq=input_seq)
        logits_tuple = outputs.logits
        pred_tuple = outputs.prediction

        loss_weight = 1.0 / float(self.num_tasks)

        loss_all = torch.tensor(0.0, device=self.device)
        batch_all = 1

        loss_task = {}
        batch_task = {}
        correct_task = {}
        samples_task = {}
        valid_count_task = {}

        per_task_losses = {}
        for t in range(self.num_tasks):
            loss_t, valid = self._masked_ce_loss(
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

        # L1 + L2 regularization
        l1_reg = torch.tensor(0.0, device=self.device)
        l2_reg = torch.tensor(0.0, device=self.device)
        for name, p in self.model.named_parameters():
            if 'omega_chol' not in name:  # Don't L1/L2 the PMR params
                l1_reg = l1_reg + torch.sum(torch.abs(p))
                l2_reg = l2_reg + torch.sum(torch.square(p))
        loss_all = loss_all + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg

        # === PMR MODIFICATION START ===
        # Add precision matrix regularization loss (after warmup)
        if self.current_epoch >= self.pmr_warmup_epochs:
            pmr_loss = self.model.get_pmr_loss()
            loss_all = loss_all + pmr_loss
        # === PMR MODIFICATION END ===

        if train_mode:
            loss_all.backward()
            self.optimizer.step()

        correct_all = 0
        samples_all = 0
        for t in range(self.num_tasks):
            c, s = self._masked_accuracy(
                pred_tuple[t], labels_list[t], ignore_index=self.ignore_index
            )
            correct_task[t] = c
            samples_task[t] = s
            correct_all += c
            samples_all += s

        self._save_debug_counts("train" if train_mode else "val", valid_count_task)

        from trainer.trainer_utils import BatchStats
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

    def _masked_ce_loss(self, logits, labels, loss_fn, ignore_index=-1):
        """Local copy of masked_ce_loss to avoid import issues."""
        mask = labels != ignore_index
        labels_masked = labels[mask]
        if labels_masked.numel() == 0:
            return None, 0
        logits_masked = logits[mask, :]
        loss = loss_fn(logits_masked, labels_masked)
        return loss, int(labels_masked.size(0))

    def _masked_accuracy(self, pred, labels, ignore_index=-1):
        """Local copy of masked_accuracy to avoid import issues."""
        mask = labels != ignore_index
        correct = int((pred[mask] == labels[mask]).sum().item())
        total = int(labels[mask].numel())
        return correct, total

    def train(self) -> None:
        logging.info(
            "training_start_PMR | task_type=%s | num_tasks=%d | epochs=%d | batch_size=%d | device=%s",
            self.task_type, self.num_tasks, self.num_epochs,
            self.batch_size, str(self.device)
        )

        for ep in range(1, self.num_epochs + 1):
            self.current_epoch = ep
            logging.info("epoch_start | epoch=%d/%d", ep, self.num_epochs)

            train_stats = self._process_data_loader(self.train_dataloader, train_mode=True)
            val_stats = self._process_data_loader(self.val_dataloader, train_mode=False)

            # Log precision matrix state
            if ep % 5 == 0:
                Omega = self.model.get_precision_matrix()
                n_off_diag = self.model.num_tasks * (self.model.num_tasks - 1)
                n_nonzero = (torch.abs(Omega - torch.diag(torch.diag(Omega))) > 1e-4).sum().item()
                logging.info(f"PMR Omega: {n_nonzero}/{n_off_diag} non-zero off-diagonal entries")

            # History tracking
            self.train_losses_all.append(train_stats.avg_loss_all)
            self.train_acc_all.append(train_stats.accuracy_all)
            self.val_losses_all.append(val_stats.avg_loss_all)
            self.val_acc_all.append(val_stats.accuracy_all)

            for t in range(self.num_tasks):
                self.train_losses_task[t].append(train_stats.avg_loss_task[t])
                self.train_acc_task[t].append(train_stats.accuracy_task[t])
                self.val_losses_task[t].append(val_stats.avg_loss_task[t])
                self.val_acc_task[t].append(val_stats.accuracy_task[t])

            # Metrics logging
            self.metrics_writer.write_metrics_all(
                self._epoch_report_line(ep, "train", train_stats))
            self.metrics_writer.write_metrics_all(
                self._epoch_report_line(ep, "val", val_stats))

            for t, task in enumerate(self.task_array):
                self.metrics_writer.write_metrics_task(
                    t,
                    f"Epoch {ep}/{self.num_epochs} | {task} | "
                    f"train_loss={train_stats.avg_loss_task[t]:.4f} "
                    f"train_acc={train_stats.accuracy_task[t]:.4f} | "
                    f"val_loss={val_stats.avg_loss_task[t]:.4f} "
                    f"val_acc={val_stats.accuracy_task[t]:.4f}"
                )

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
                    "epoch_task | epoch=%d | task=%s | train(loss=%.4f acc=%.4f) | "
                    "val(loss=%.4f acc=%.4f) | samples=%d",
                    ep, task,
                    train_stats.avg_loss_task[t], train_stats.accuracy_task[t],
                    val_stats.avg_loss_task[t], val_stats.accuracy_task[t],
                    int(val_stats.total_samples_task.get(t, 0))
                )

            # Checkpoints + scheduler
            self.ckpt.save_epoch_checkpoint(ep)
            self.ckpt.maybe_save_best_and_opt(
                test_accuracy_all=val_stats.accuracy_all,
                test_accuracy_task=val_stats.accuracy_task,
                num_tasks=self.num_tasks
            )
            self.scheduler.step(val_stats.avg_loss_all)

        # Final summary
        self.metrics_writer.write_metrics_all(
            f"Best model accuracy - all: {self.ckpt.best_accuracy_all_threshold:.6f}")
        for t, task in enumerate(self.task_array):
            self.metrics_writer.write_metrics_all(
                f"Best model accuracy - {task}: {self.ckpt.best_accuracy_task_thresholds.get(t, 0.0):.6f}"
            )
        self.metrics_writer.write_metrics_all(
            f"Opt model accuracy - all: {self.ckpt.opt_accuracy_all_threshold:.6f}")
        for t, task in enumerate(self.task_array):
            self.metrics_writer.write_metrics_all(
                f"Opt model accuracy - {task}: {self.ckpt.opt_accuracy_task_thresholds.get(t, 0.0):.6f}"
            )

        logging.info("training_end | run_id=%s | best_all=%.6f | opt_all=%.6f",
                     os.path.basename(self.results_dir),
                     self.ckpt.best_accuracy_all_threshold,
                     self.ckpt.opt_accuracy_all_threshold)

        self.plot_metrics()
