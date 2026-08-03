"""
Downstream multi task classification model.

This module defines a shared representation based downstream
multi task learning model that operates on pre computed
upstream embeddings. It supports:
- Layer wise pooling across transformer layers
- Shared projection and hidden layers
- Multiple task specific classification heads
- Retrieval of pooled embeddings and pooling weights

The model is designed to be used with upstream SSL
representations such as WavLM.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pooling.pooling import Pooling
from utils.constant_mapping import LabelKeywordMapping, TaskDatasetMapping
from utils.pooling_id import make_pooling_name

class MultiClassifierOutput:
    def __init__(self, logits=None, prediction=None):
        self.logits = logits
        self.prediction = prediction


class DownstreamMultiTaskModel(nn.Module):
    def __init__(
        self,
        upstream_model_type: str,
        task_type: str,
        embedding_dim_shared1: int,
        embedding_dim_shared2: int,
        layer_pooling_type: str,
        dropout_prob_shared1: float,
        dropout_prob_shared2: float,
        layer_pooling_param: Optional[Union[int, float]] = None
    ):
        super().__init__()
        
        input_dim = self._input_dim_from_upstream(upstream_model_type)
        output_dim_array = self._output_dims_from_task_type(task_type)

        self.projector_layer = nn.Linear(input_dim, embedding_dim_shared1)
        self.dropout_shared1 = nn.Dropout(p=dropout_prob_shared1)

        self.hidden_layer = nn.Linear(embedding_dim_shared1, embedding_dim_shared2)
        self.dropout_shared2 = nn.Dropout(p=dropout_prob_shared2)

        self.classifiers = nn.ModuleList([
            nn.Linear(embedding_dim_shared2, out_dim) for out_dim in output_dim_array
        ])

        # Pooling owns learnable parameters only when needed (weighted)
        self.layer_pooling_type = layer_pooling_type
        self.pooling = Pooling(layer_pooling_type, pooling_param=layer_pooling_param)
        
        layer_pool_name = make_pooling_name(layer_pooling_type, layer_pooling_param)
        logging.info(f"Layer pooling type: {layer_pool_name}")
        
    def _input_dim_from_upstream(self, upstream_model_type: str) -> int:
        upstream_model_variation = upstream_model_type.split("_")[-1].lower()
        if upstream_model_variation == "base":
            return 768
        elif upstream_model_variation == "large":
            return 1024
        else:
            raise ValueError(
                f"Unknown upstream_model_variation='{upstream_model_variation}' "
                f"from upstream_model_type='{upstream_model_type}'. Expected 'base' or 'large'."
            )
            
    def _build_dataset_id_from_task_type(self, task_type: str) -> list[str]:
        task_tokens = task_type.split("_")
        dataset_keys = []
        for t in task_tokens:
            ds = TaskDatasetMapping.get_dataset_key(t)
            if ds is None:
                raise ValueError(f"Invalid task type token: '{t}' in task_type='{task_type}'")
            if ds not in dataset_keys:
                dataset_keys.append(ds)
        return dataset_keys

    def _output_dims_from_task_type(self, task_type: str) -> list[int]:
        dataset_id_array = self._build_dataset_id_from_task_type(task_type)

        output_dim_array: list[int] = []
        for ds in dataset_id_array:
            label2index, _ = LabelKeywordMapping.get_label_mapping(ds)
            output_dim_array.append(len(label2index))

        return output_dim_array

    def forward(self, input_seq):
        # input_seq: [B, L, input_dim] e.g., [2048, 25, 1024]
        embedding_shared = self.projector_layer(input_seq)  # [B, L, shared1]
        embedding_shared = self.pooling.get_vector_after_pooling(embedding_shared, dim=1)  # [B, shared1]
        embedding_shared = self.dropout_shared1(embedding_shared)

        embedding_shared = self.hidden_layer(embedding_shared)  # [B, shared2]
        embedding_shared = self.dropout_shared2(embedding_shared)

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
