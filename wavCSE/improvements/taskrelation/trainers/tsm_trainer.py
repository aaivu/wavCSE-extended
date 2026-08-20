"""
TSM Trainer: Alternating minimization for wavCSE-TSM.

Inspired by Ciliberto et al. (2015) Algorithm 1:
- Supervised step: train model parameters (with structure matrix A fixed)
- Unsupervised step: optimize structure matrix A (with model fixed)

The alternating schedule:
- Every `alternate_frequency` epochs, we freeze the model backbone
  and optimize only the structure matrix A for `structure_epochs` iterations.
"""

import os
import logging
import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../downstream'))

from trainer.trainer_model import MultiTasksModelTrainer


class MultiTasksModelTrainerTSM(MultiTasksModelTrainer):
    """
    Trainer for wavCSE-TSM with alternating minimization.

    Extends the standard trainer with:
    - Structure matrix A optimization phase
    - Combined loss: task_loss + sparsity_loss(A)
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
        alternate_frequency: int = 5,     # How often to alternate (epochs)
        structure_epochs: int = 1,        # How many structure update epochs
        structure_lr: float = 0.001,      # Learning rate for structure matrix
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

        self.alternate_frequency = alternate_frequency
        self.structure_epochs = structure_epochs
        self.structure_lr = structure_lr

        # Create a separate optimizer for the structure matrix A
        structure_params = [self.model.structure_matrix_A]
        self.structure_optimizer = torch.optim.Adam(
            structure_params, lr=structure_lr
        )

        logging.info(f"TSM alternating: frequency={alternate_frequency}, "
                     f"structure_epochs={structure_epochs}, structure_lr={structure_lr}")

    def _optimize_structure_matrix(self):
        """
        Unsupervised step: optimize structure matrix A with model frozen.

        Following Ciliberto et al. (2015), the structure matrix is optimized
        to minimize: sparsity_loss(A) = lambda * |A|_1 + ||A - I_proj||_F^2
        where I_proj encourages A to be near-identity for stability.

        We use the training data to get meaningful gradients through the
        classification loss with respect to A.
        """
        self.model.train()
        # Freeze all parameters except structure matrix
        for name, param in self.model.named_parameters():
            if 'structure_matrix_A' not in name:
                param.requires_grad = False

        for _ in range(self.structure_epochs):
            for batch in self.train_dataloader:
                input_seq, labels_list = self._unpack_batch(batch)
                self.structure_optimizer.zero_grad(set_to_none=True)

                outputs = self.model(input_seq=input_seq)
                logits_tuple = outputs.logits

                # Compute classification loss through A
                loss_weight = 1.0 / float(self.num_tasks)
                total_loss = torch.tensor(0.0, device=self.device)
                for t in range(self.num_tasks):
                    mask = labels_list[t] != self.ignore_index
                    if mask.sum() == 0:
                        continue
                    loss_t = self.loss_fn(
                        logits_tuple[t][mask], labels_list[t][mask]
                    )
                    total_loss = total_loss + loss_t * loss_weight

                # Add structure sparsity loss
                sparsity_loss = self.model.get_sparsity_loss()
                total_loss = total_loss + sparsity_loss

                total_loss.backward()
                self.structure_optimizer.step()

        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        logging.info(f"Structure matrix A updated. Sparsity: "
                     f"{(self.model.structure_matrix_A.abs() < 1e-4).float().mean().item():.3f}")

    def train(self) -> None:
        logging.info(
            "training_start_TSM | task_type=%s | num_tasks=%d | epochs=%d | batch_size=%d | device=%s",
            self.task_type, self.num_tasks, self.num_epochs,
            self.batch_size, str(self.device)
        )

        for ep in range(1, self.num_epochs + 1):
            logging.info("epoch_start | epoch=%d/%d", ep, self.num_epochs)

            # Standard supervised training epoch
            train_stats = self._process_data_loader(self.train_dataloader, train_mode=True)
            val_stats = self._process_data_loader(self.val_dataloader, train_mode=False)

            # Alternating minimization: periodically optimize structure matrix
            if ep % self.alternate_frequency == 0:
                logging.info(f"Alternating: optimizing structure matrix A at epoch {ep}")
                self._optimize_structure_matrix()

            # History tracking (same as original)
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
