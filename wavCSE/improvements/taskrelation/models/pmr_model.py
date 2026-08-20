"""
wavCSE-PMR: Precision Matrix Regularization (inspired by Goncalves et al. 2016)

Goncalves et al. (2016) proposed Multi-task Sparse Structure Learning (MSSL),
which learns a sparse precision (inverse covariance) matrix Omega over task
parameters. Key ideas:
1. The rows of the parameter matrix W follow a multivariate Gaussian with
   precision matrix Omega: w_j ~ N(0, Omega^{-1})
2. Omega encodes conditional independence: Omega_ij = 0 means tasks i, j
   are conditionally independent given all other tasks
3. The precision matrix is learned jointly with task parameters via
   alternating minimization
4. l1 regularization on Omega enforces sparsity -> interpretable task graphs
5. The loss includes: -log|Omega| + Tr(W * Omega * W^T) + lambda * |Omega|_1

Two variants from the paper:
- p-MSSL: models precision over task parameters W
- r-MSSL: models precision over residual errors (regression only)

We implement p-MSSL (applicable to both classification and regression).

Applied to wavCSE:
- We collect the final-layer task head weights into a matrix W of shape
  [num_tasks, embedding_dim_shared2 * num_classes_avg].
- We learn a sparse precision matrix Omega over tasks (shape [T, T]).
- The loss gets an additional term encouraging task structure:
  -log|Omega| + Tr(W_flat * Omega * W_flat^T) + lambda_sparse * |Omega|_1
- Omega_ij = 0 indicates tasks i and j are conditionally independent.

Key difference from TSM (Ciliberto): PMR uses a PRECISION matrix (inverse
covariance) which has a direct graphical model interpretation as conditional
independence, while TSM uses a structure matrix A for direct task coupling.
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


class DownstreamMultiTaskModelPMR(nn.Module):
    """
    wavCSE with Precision Matrix Regularization (PMR).

    Keeps independent task heads but adds a learnable sparse precision
    matrix Omega that regularizes the relationships between task parameters.

    Omega is a [num_tasks, num_tasks] PSD matrix where:
    - Omega_ij != 0: tasks i and j are conditionally dependent
    - Omega_ij = 0:  tasks i and j are conditionally independent
    - Diagonal: task-specific precision (inverse variance)
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
        pmr_lambda: float = 0.01,    # NEW: sparsity penalty on Omega
        pmr_gamma: float = 0.1,      # NEW: strength of precision matrix regularization
        layer_pooling_param: Optional[Union[int, float]] = None
    ):
        super().__init__()

        input_dim = self._input_dim_from_upstream(upstream_model_type)
        output_dim_array = self._output_dims_from_task_type(task_type)
        self.num_tasks = len(output_dim_array)
        self.pmr_lambda = pmr_lambda
        self.pmr_gamma = pmr_gamma

        # Shared backbone (unchanged from wavCSE)
        self.projector_layer = nn.Linear(input_dim, embedding_dim_shared1)
        self.dropout_shared1 = nn.Dropout(p=dropout_prob_shared1)

        self.hidden_layer = nn.Linear(embedding_dim_shared1, embedding_dim_shared2)
        self.dropout_shared2 = nn.Dropout(p=dropout_prob_shared2)

        # Task-specific heads (unchanged structure)
        self.classifiers = nn.ModuleList([
            nn.Linear(embedding_dim_shared2, out_dim) for out_dim in output_dim_array
        ])

        # === PMR MODIFICATION START ===
        # Precision matrix Omega: [num_tasks, num_tasks]
        # Parameterized via its Cholesky factor L to ensure PSD constraint
        # Omega = L @ L^T where L is lower triangular
        self.omega_chol = nn.Parameter(
            torch.eye(self.num_tasks) * 0.1
        )
        # === PMR MODIFICATION END ===

        self.layer_pooling_type = layer_pooling_type
        self.pooling = Pooling(layer_pooling_type, pooling_param=layer_pooling_param)

        layer_pool_name = make_pooling_name(layer_pooling_type, layer_pooling_param)
        logging.info(f"Layer pooling type: {layer_pool_name}")
        logging.info(f"PMR lambda (sparsity): {pmr_lambda}")
        logging.info(f"PMR gamma (strength): {pmr_gamma}")

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

    def get_precision_matrix(self):
        """Return the learned precision matrix Omega for analysis."""
        L = torch.tril(self.omega_chol)
        Omega = L @ L.T
        return Omega.detach().cpu()

    def get_task_parameter_matrix(self):
        """
        Collect task head weights into a matrix for the trace term.

        Each task head has weight [out_dim, hidden_dim] and bias [out_dim].
        We flatten each head's parameters and stack them into W: [num_tasks, D].
        """
        rows = []
        for head in self.classifiers:
            w = head.weight.data.flatten()  # [out_dim * hidden_dim]
            b = head.bias.data              # [out_dim]
            row = torch.cat([w, b])         # [out_dim * hidden_dim + out_dim]
            rows.append(row)
        W = torch.stack(rows, dim=0)  # [num_tasks, total_params_per_task]
        return W

    def get_pmr_loss(self):
        """
        Compute the precision matrix regularization loss.

        Loss = gamma * [ -log|Omega| + Tr(W * Omega * W^T) ] + lambda * |Omega|_1

        - -log|Omega|: encourages well-conditioned precision (prevents collapse)
        - Tr(W * Omega * W^T): couples task parameters through precision matrix
        - lambda * |Omega|_1: sparsity on task relationships

        Returns:
            pmr_loss: scalar tensor
        """
        # Reconstruct Omega from Cholesky factor (ensures PSD)
        L = torch.tril(self.omega_chol)
        Omega = L @ L.T  # [num_tasks, num_tasks]

        # Log-determinant term: -log|Omega|
        # log|Omega| = 2 * sum(log(diag(L)))
        log_det = 2.0 * torch.sum(torch.log(torch.abs(torch.diag(L)) + 1e-10))
        neg_log_det = -log_det

        # Trace term: Tr(W * Omega * W^T)
        W = self.get_task_parameter_matrix()
        # W: [num_tasks, D], Omega: [num_tasks, num_tasks]
        W_omega = W.T @ Omega  # [D, num_tasks]
        trace_term = torch.trace(W_omega @ W)  # = Tr(W^T * Omega * W)

        # Combined PMR loss
        pmr_loss = self.pmr_gamma * (neg_log_det + trace_term)

        # Sparsity penalty on off-diagonal entries of Omega
        off_diag_mask = 1.0 - torch.eye(self.num_tasks, device=Omega.device)
        sparsity_penalty = self.pmr_lambda * torch.sum(
            torch.abs(Omega * off_diag_mask)
        )

        return pmr_loss + sparsity_penalty

    def forward(self, input_seq):
        # Shared backbone (unchanged)
        embedding_shared = self.projector_layer(input_seq)
        embedding_shared = self.pooling.get_vector_after_pooling(embedding_shared, dim=1)
        embedding_shared = self.dropout_shared1(embedding_shared)

        embedding_shared = self.hidden_layer(embedding_shared)
        embedding_shared = self.dropout_shared2(embedding_shared)

        # Task heads (unchanged forward pass)
        logits_list = []
        pred_list = []
        for head in self.classifiers:
            logits = head(embedding_shared)
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
