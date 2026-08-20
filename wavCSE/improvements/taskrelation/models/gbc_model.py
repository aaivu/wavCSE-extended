"""
wavCSE-GBC: Global Bias Coupling (inspired by Zhang et al. 2010)

Zhang et al. (2010) proposed Bayesian Online Multi-label Classification (BOMC)
where a shared global bias parameter couples all classes in a multi-label setting.
The key insight is that a shared hub parameter (global bias) serves as a coupling
mechanism: it gets updated more often than per-class parameters, leading to more
confident estimates and better information sharing across tasks.

Applied to wavCSE:
- Instead of independent task classification heads with unconstrained biases,
  we introduce a learnable global bias vector b_global that is shared across
  all task heads.
- Each task head output becomes: logits_t = W_t * h + b_t + b_global
  where b_global couples all tasks through a common reference point.
- This models the intuition that task base rates are related (e.g., if one
  task has a high base rate for a class, related tasks may too).

This is the LIGHTEST change to wavCSE - only modifies the classifier heads.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from parent wavCSE (adjusted for sys.path in run script)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../downstream'))

from pooling.pooling import Pooling
from utils.constant_mapping import LabelKeywordMapping, TaskDatasetMapping
from utils.pooling_id import make_pooling_name


class MultiClassifierOutput:
    def __init__(self, logits=None, prediction=None):
        self.logits = logits
        self.prediction = prediction


class DownstreamMultiTaskModelGBC(nn.Module):
    """
    wavCSE with Global Bias Coupling (GBC).

    Adds a shared global bias that couples all task classification heads,
    inspired by the BOMC framework (Zhang et al., 2010).

    The global bias acts as a "hub" parameter - updated by every training
    example regardless of task, leading to more confident estimates and
    better information sharing.
    """

    def __init__(
        self,
        upstream_model_type: str,
        task_type: str,
        embedding_dim_shared1: int,
        embedding_dim_shared2: int,
        layer_pooling_type: str,
        dropout_prob_shared1: float,
        dropout_prob_shared2: float,
        gbc_global_dim: int = 64,  # NEW: dimension of global bias embedding
        layer_pooling_param: Optional[Union[int, float]] = None
    ):
        super().__init__()

        input_dim = self._input_dim_from_upstream(upstream_model_type)
        output_dim_array = self._output_dims_from_task_type(task_type)
        self.num_tasks = len(output_dim_array)
        self.gbc_global_dim = gbc_global_dim

        # Shared backbone (unchanged from wavCSE)
        self.projector_layer = nn.Linear(input_dim, embedding_dim_shared1)
        self.dropout_shared1 = nn.Dropout(p=dropout_prob_shared1)

        self.hidden_layer = nn.Linear(embedding_dim_shared1, embedding_dim_shared2)
        self.dropout_shared2 = nn.Dropout(p=dropout_prob_shared2)

        # === GBC MODIFICATION START ===
        # Global bias module: projects the shared representation to a lower-dim
        # "global bias space" that couples all tasks
        self.global_bias_projector = nn.Linear(embedding_dim_shared2, gbc_global_dim)
        self.global_bias = nn.Parameter(torch.zeros(gbc_global_dim))

        # Per-task heads: each task gets its own weight matrix, but the bias
        # is influenced by both a per-task learned bias AND the global coupling bias
        self.classifiers = nn.ModuleList([
            nn.Linear(embedding_dim_shared2, out_dim)
            for out_dim in output_dim_array
        ])
        # Per-task learned bias vectors (for task-specific bias adjustment)
        self.task_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(out_dim))
            for out_dim in output_dim_array
        ])
        # Per-task projection from global bias to task output space
        self.global_to_task_bias = nn.ModuleList([
            nn.Linear(gbc_global_dim, out_dim, bias=False)
            for out_dim in output_dim_array
        ])
        # === GBC MODIFICATION END ===

        self.layer_pooling_type = layer_pooling_type
        self.pooling = Pooling(layer_pooling_type, pooling_param=layer_pooling_param)

        layer_pool_name = make_pooling_name(layer_pooling_type, layer_pooling_param)
        logging.info(f"Layer pooling type: {layer_pool_name}")
        logging.info(f"GBC global bias dim: {gbc_global_dim}")

    def _input_dim_from_upstream(self, upstream_model_type: str) -> int:
        upstream_model_variation = upstream_model_type.split("_")[-1].lower()
        if upstream_model_variation == "base":
            return 768
        elif upstream_model_variation == "large":
            return 1024
        else:
            raise ValueError(
                f"Unknown upstream_model_variation='{upstream_model_variation}' "
                f"from upstream_model_type='{upstream_model_type}'."
            )

    def _build_dataset_id_from_task_type(self, task_type: str) -> list:
        task_tokens = task_type.split("_")
        dataset_keys = []
        for t in task_tokens:
            ds = TaskDatasetMapping.get_dataset_key(t)
            if ds is None:
                raise ValueError(f"Invalid task type token: '{t}' in task_type='{task_type}'")
            if ds not in dataset_keys:
                dataset_keys.append(ds)
        return dataset_keys

    def _output_dims_from_task_type(self, task_type: str) -> list:
        dataset_id_array = self._build_dataset_id_from_task_type(task_type)
        output_dim_array = []
        for ds in dataset_id_array:
            label2index, _ = LabelKeywordMapping.get_label_mapping(ds)
            output_dim_array.append(len(label2index))
        return output_dim_array

    def forward(self, input_seq):
        # Shared backbone (unchanged)
        embedding_shared = self.projector_layer(input_seq)
        embedding_shared = self.pooling.get_vector_after_pooling(embedding_shared, dim=1)
        embedding_shared = self.dropout_shared1(embedding_shared)

        embedding_shared = self.hidden_layer(embedding_shared)
        embedding_shared = self.dropout_shared2(embedding_shared)

        # === GBC FORWARD ===
        # Compute the global coupling bias from the shared representation
        global_bias_proj = self.global_bias_projector(embedding_shared)
        # Combine: learned global bias + representation-dependent coupling
        global_coupling = self.global_bias + global_bias_proj  # [B, gbc_global_dim]

        logits_list = []
        pred_list = []
        for t, head in enumerate(self.classifiers):
            # Base logits from task-specific weight matrix (no bias from Linear)
            logits = head(embedding_shared)  # weight * x (bias handled separately)
            # Add task-specific learned bias
            logits = logits + self.task_biases[t]
            # Add global coupling contribution (maps global bias to task output space)
            logits = logits + self.global_to_task_bias[t](global_coupling)
            pred = torch.argmax(logits, dim=1)
            logits_list.append(logits)
            pred_list.append(pred)

        return MultiClassifierOutput(
            logits=tuple(logits_list),
            prediction=tuple(pred_list)
        )

    def get_all_embeddings(self, input_seq):
        embedding_shared = self.projector_layer(input_seq)
        embedding_shared = self.pooling.get_vector_after_pooling(embedding_shared, dim=1)
        embedding_shared = self.hidden_layer(embedding_shared)
        return embedding_shared

    def get_pooling_weights(self):
        if self.layer_pooling_type != "weighted":
            return None
        return F.softmax(self.pooling.pooling_weights, dim=0)

    def get_global_bias_coupling(self):
        """Return the learned global bias for analysis."""
        return self.global_bias.detach().cpu()
