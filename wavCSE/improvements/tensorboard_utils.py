"""Reusable TensorBoard instrumentation for trainers under ``improvements``."""

import logging
import os
from typing import Type

import torch


class TensorBoardTrainerMixin:
    """Add TensorBoard logging without changing the downstream trainer.

    Put this mixin before a trainer class in the MRO.  Logging is enabled by
    default and event files are kept with the trainer's other run artifacts.
    """

    def __init__(self, *args, tensorboard_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = tensorboard_cfg or {}
        self.tensorboard_enabled = bool(cfg.get("enabled", True))
        self._tb_writer = None
        self._tb_step = 0
        self._tb_capture_gradients = False
        self._tb_gradient_interval = max(1, int(cfg.get("gradient_log_interval", 100)))
        self._tb_histogram_interval = max(1, int(cfg.get("histogram_epoch_interval", 5)))
        self._tb_gradient_norms = {}
        self._tb_previous_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self._tb_hook_handles = []

        if not self.tensorboard_enabled:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logging.warning(
                "TensorBoard is enabled but the 'tensorboard' package is not installed; "
                "continuing without TensorBoard logging."
            )
            self.tensorboard_enabled = False
            return

        configured_dir = cfg.get("log_dir")
        log_dir = (
            os.path.expanduser(configured_dir)
            if configured_dir
            else os.path.join(self.results_dir, "tensorboard")
        )
        self._tb_writer = SummaryWriter(log_dir=log_dir)
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self._tb_hook_handles.append(
                    parameter.register_hook(self._make_gradient_hook(name))
                )
        logging.info("TensorBoard logging enabled: %s", log_dir)

    def _make_gradient_hook(self, name):
        def record(gradient):
            if self._tb_capture_gradients:
                self._tb_gradient_norms[name] = float(gradient.detach().norm().item())
            return gradient

        return record

    def _process_batch(self, batch, train_mode):
        self._tb_capture_gradients = bool(
            self.tensorboard_enabled
            and train_mode
            and self._tb_step % self._tb_gradient_interval == 0
        )
        stats = super()._process_batch(batch, train_mode=train_mode)
        if train_mode:
            if self._tb_capture_gradients and self._tb_writer is not None:
                for name, norm in self._tb_gradient_norms.items():
                    self._tb_writer.add_scalar(f"gradient_norm/{name}", norm, self._tb_step)
            self._tb_gradient_norms.clear()
            self._tb_step += 1
        self._tb_capture_gradients = False
        return stats

    def _epoch_report_line(self, epoch, phase, stats):
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(f"loss/{phase}/all", stats.avg_loss_all, epoch)
            self._tb_writer.add_scalar(f"accuracy/{phase}/all", stats.accuracy_all, epoch)
            for index, task in enumerate(self.task_array):
                self._tb_writer.add_scalar(
                    f"loss/{phase}/{task}", stats.avg_loss_task[index], epoch
                )
                self._tb_writer.add_scalar(
                    f"accuracy/{phase}/{task}", stats.accuracy_task[index], epoch
                )
            if phase == "val":
                self._tb_writer.add_scalar(
                    "optimizer/learning_rate",
                    float(self.optimizer.param_groups[0]["lr"]),
                    epoch,
                )
                self._log_tensorboard_parameters(epoch)
                self._tb_writer.flush()
        return super()._epoch_report_line(epoch, phase, stats)

    def _log_tensorboard_parameters(self, epoch):
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            current = parameter.detach().cpu()
            previous = self._tb_previous_parameters.get(name)
            parameter_norm = float(current.norm().item())
            self._tb_writer.add_scalar(f"parameter_norm/{name}", parameter_norm, epoch)
            if previous is not None:
                change_norm = float((current - previous).norm().item())
                self._tb_writer.add_scalar(f"parameter_change/{name}", change_norm, epoch)
                self._tb_writer.add_scalar(
                    f"relative_parameter_change/{name}",
                    change_norm / (parameter_norm + 1e-12),
                    epoch,
                )
            if epoch == 1 or epoch % self._tb_histogram_interval == 0:
                self._tb_writer.add_histogram(f"parameters/{name}", current, epoch)
            self._tb_previous_parameters[name] = current.clone()

    def train(self):
        try:
            return super().train()
        finally:
            if self._tb_writer is not None:
                self._tb_writer.close()
            for handle in self._tb_hook_handles:
                handle.remove()


def tensorboard_trainer(trainer_class: Type) -> Type:
    """Return a trainer class instrumented with :class:`TensorBoardTrainerMixin`."""
    return type(
        f"TensorBoard{trainer_class.__name__}",
        (TensorBoardTrainerMixin, trainer_class),
        {"__module__": trainer_class.__module__},
    )
