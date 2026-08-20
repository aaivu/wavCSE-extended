"""
wavCSE-TSM: Task Structure Matrix (inspired by Ciliberto et al. 2015)

Ciliberto et al. (2015) proposed Sparse Kernel MTL, which learns a sparse
task relationship matrix A that encodes how tasks relate to each other.
Key ideas:
1. Each task predictor f_t is a linear combination of latent basis functions g_s
   via a structure matrix A: f_t = sum_s A_ts * g_s
2. The structure matrix A is learned with sparsity (l1) regularization to recover
   only the most relevant task relations
3. Alternating minimization: supervise step (learn f given A), then unsupervised
   step (learn A given f)
4. Formal convergence guarantees via convex optimization

Applied to wavCSE:
- Instead of independent task heads, we introduce a set of K "latent classifiers"
  (g_1, ..., g_K) that form a basis in classifier space.
- Each actual task head is a sparse linear combination of these latent classifiers
  through a learned structure matrix A ∈ R^{T × K}.
- The structure matrix A is regularized with l1 to discover only the most
  relevant task relations (prevents negative transfer).
- We use alternating training: optimize model with fixed A, then optimize A
  given the model (using a simple gradient-based approach).

The Recursive Feature Selection (layer selection + pooling) is PRESERVED.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DownstreamMultiTaskModelTSM(nn.Module):
    """
    wavCSE with Task Structure Matrix (TSM).

    Replaces independent task heads with a learned latent classifier basis
    and a sparse task structure matrix A that maps latent classifiers to
    actual task outputs.

    Architecture change:
      Original:  shared_emb -> [head_1, head_2, ..., head_T]  (independent)
      TSM:       shared_emb -> [latent_1, ..., latent_K] -> A -> task outputs
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
        num_latent_classifiers: int = 8,  # NEW: K = number of latent classifiers
        structure_sparsity_lambda: float = 0.01,  # NEW: l1 penalty on A
        layer_pooling_param: Optional[Union[int, float]] = None
    ):
        super().__init__()

        input_dim = self._input_dim_from_upstream(upstream_model_type)
        output_dim_array = self._output_dims_from_task_type(task_type)
        self.num_tasks = len(output_dim_array)
        self.num_latent = num_latent_classifiers
        self.output_dim_array = output_dim_array
        self.structure_sparsity_lambda = structure_sparsity_lambda

        # Shared backbone (unchanged from wavCSE)
        self.projector_layer = nn.Linear(input_dim, embedding_dim_shared1)
        self.dropout_shared1 = nn.Dropout(p=dropout_prob_shared1)

        self.hidden_layer = nn.Linear(embedding_dim_shared1, embedding_dim_shared2)
        self.dropout_shared2 = nn.Dropout(p=dropout_prob_shared2)

        # === TSM MODIFICATION START ===
        # Latent classifier basis: K linear classifiers that form a basis
        # Each latent classifier maps the shared embedding to a "latent logit space"
        # of dimension max(output_dims) for simplicity (we handle varying dims via slicing)
        self.max_output_dim = max(output_dim_array)
        self.latent_classifiers = nn.ModuleList([
            nn.Linear(embedding_dim_shared2, self.max_output_dim)
            for _ in range(num_latent_classifiers)
        ])

        # Task structure matrix A: maps latent classifiers -> actual tasks
        # A[t, k] = weight of latent classifier k for task t
        # Learnable parameter with l1 regularization for sparsity
        self.structure_matrix_A = nn.Parameter(
            torch.randn(self.num_tasks, num_latent_classifiers) * 0.01
        )

        # Per-task bias (independent, since biases don't carry structural info)
        self.task_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(out_dim))
            for out_dim in output_dim_array
        ])
        # === TSM MODIFICATION END ===

        self.layer_pooling_type = layer_pooling_type
        self.pooling = Pooling(layer_pooling_type, pooling_param=layer_pooling_param)

        layer_pool_name = make_pooling_name(layer_pooling_type, layer_pooling_param)
        logging.info(f"Layer pooling type: {layer_pool_name}")
        logging.info(f"TSM num_latent_classifiers: {num_latent_classifiers}")
        logging.info(f"TSM structure_sparsity_lambda: {structure_sparsity_lambda}")

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

    def get_structure_matrix(self):
        """Return the learned task structure matrix for analysis/visualization."""
        return self.structure_matrix_A.detach().cpu()

    def get_sparsity_loss(self):
        """l1 penalty on the structure matrix to encourage sparse task relations."""
        return self.structure_sparsity_lambda * torch.sum(torch.abs(self.structure_matrix_A))

    def forward(self, input_seq):
        # Shared backbone (unchanged)
        embedding_shared = self.projector_layer(input_seq)
        embedding_shared = self.pooling.get_vector_after_pooling(embedding_shared, dim=1)
        embedding_shared = self.dropout_shared1(embedding_shared)

        embedding_shared = self.hidden_layer(embedding_shared)
        embedding_shared = self.dropout_shared2(embedding_shared)

        # === TSM FORWARD ===
        # Step 1: Compute all latent classifier outputs
        # Each latent classifier produces a [B, max_output_dim] tensor
        latent_outputs = []
        for k in range(self.num_latent):
            latent_k = self.latent_classifiers[k](embedding_shared)  # [B, max_dim]
            latent_outputs.append(latent_k)
        # Stack: [B, K, max_output_dim]
        latent_stack = torch.stack(latent_outputs, dim=1)

        # Step 2: Apply structure matrix A to combine latent classifiers
        # A: [T, K], softmax over K for each task to make it a convex combination
        A_soft = F.softmax(self.structure_matrix_A, dim=1)  # [T, K]

        logits_list = []
        pred_list = []
        for t in range(self.num_tasks):
            out_dim_t = self.output_dim_array[t]
            # Weighted combination of latent classifiers for task t
            # A_soft[t]: [K], latent_stack: [B, K, max_dim]
            # -> task_logits: [B, max_dim]
            task_logits_full = torch.einsum('bkd,k->bd', latent_stack, A_soft[t])
            # Slice to the correct output dimension for this task
            task_logits = task_logits_full[:, :out_dim_t]
            # Add per-task bias
            task_logits = task_logits + self.task_biases[t]

            pred = torch.argmax(task_logits, dim=1)
            logits_list.append(task_logits)
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
