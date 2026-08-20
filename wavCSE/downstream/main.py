"""
Main entry script for downstream multi task training + evaluation.

This script:
1. Loads configuration from YAML
2. Selects device (CPU / GPU)
3. Loads pre computed embeddings for the requested task set
4. Builds the downstream multi task model
5. Trains the model and saves checkpoints
6. Evaluates three checkpoint variants on the test split:
   - opt  : checkpoint chosen by your CheckpointManager as the "optimal" model
   - best : checkpoint chosen by your CheckpointManager as the "best" model
   - epoch: a specific epoch checkpoint

This file should stay as orchestration only.
All heavy logic remains inside dataset/, model/, trainer/, evaluator/, utils/.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import torch
from torch import Tensor

import os
import sys
import argparse

from dataset.load_embedding import LoadEmbedding
from model.downstream_model import DownstreamMultiTaskModel
from trainer.trainer_model import MultiTasksModelTrainer
from evaluator.evaluator_model import MultiTasksModelEvaluator
from utils.load_config import load_config
from utils.setup_device import set_device
from utils.setup_logging import setup_logging
from utils.parse_transformer_layers import parse_transformer_layers


# ----------------------------
# NOTE FOR NOTEBOOK USERS
# ----------------------------
# Uncomment this block if running inside Jupyter / IPython
# sys.argv = [
#     "notebook",
#     "--task_type", "ks_si_er_ic",
#     "--config", "configs/build_model.yml",
#     "--device_index", "0"
# ]


# ----------------------------
# Argument parsing (CLI only)
# ----------------------------
parser = argparse.ArgumentParser(
    description="Train and evaluate a downstream multi task model"
)

parser.add_argument(
    "--task_type",
    type=str,
    required=True,
    help="Task string joined by underscores (e.g., ks_si_er_ic)"
)

parser.add_argument(
    "--config",
    type=str,
    default="configs/build_model.yml",
    required=True,
    help="Path to YAML configuration file"
)

parser.add_argument(
    "--device_index",
    type=int,
    default=None,
    help="GPU index to use (overrides config file)"
)

args = parser.parse_args()


# ----------------------------
# Load configuration
# ----------------------------
cfg = load_config(args.config)

log_level = cfg["log_level"]
task_type = args.task_type

# Device config
device_type = cfg["device"]["type"]
device_index = (
    args.device_index
    if args.device_index is not None
    else cfg["device"]["index"]
)

# Paths config
root_data_path = cfg["paths"]["root_data_path"]
root_emb_path = cfg["paths"]["root_emb_path"]
results_root=cfg["paths"]["results_root"]
checkpoints_root=cfg["paths"]["checkpoints_root"]

# Upstream representation settings
upstream_model_type = cfg["upstream"]["model_type"]
selected_transformer_layers = cfg["upstream"]["selected_transformer_layers"]

# Parse which transformer layers to load from stored embeddings
transformer_layer_array = parse_transformer_layers(
    transformer_layers=selected_transformer_layers,
    upstream_model_type=upstream_model_type
)

# Dataset settings
subset_percentage = cfg["dataset"]["subset_percentage"]
ignore_index = cfg["dataset"]["ignore_index"]

# Pooling settings
frame_pooling_type = cfg["pooling"]["frame_pooling_type"]
layer_pooling_type = cfg["pooling"]["layer_pooling_type"]
frame_pooling_param = cfg["pooling"]["frame_pooling_param"]

# Some layer pooling needs a parameter:
# - weighted / gated need seq_len (number of selected layers)
# - others use config value directly (or null)
layer_pooling_param = (
    cfg["pooling"]["layer_pooling_param"]
    if layer_pooling_type not in ["weighted", "gated"]
    else len(transformer_layer_array)
)

# Model hyper parameters
embedding_dim_shared1 = cfg["model"]["embedding_dim_shared1"]
embedding_dim_shared2 = cfg["model"]["embedding_dim_shared2"]
dropout_prob_shared1 = cfg["model"]["dropout_prob_shared1"]
dropout_prob_shared2 = cfg["model"]["dropout_prob_shared2"]

# Training and evaluation configs
training_cfg = cfg["training"]
evaluation_cfg = cfg["evaluation"]


# ----------------------------
# Setup logging
# ----------------------------
setup_logging(log_level=log_level)


# ----------------------------
# Setup device
# ----------------------------
device = set_device(device_type=device_type, device_index=device_index)


# ----------------------------
# Load embeddings from disk
# ----------------------------
# LoadEmbedding should:
# - locate correct embedding folder based on upstream_model_type and frame_pooling_type
# - load embedding tensors
# - create train / val / test datasets
loader = LoadEmbedding(
    root_data_path=root_data_path,
    root_emb_path=root_emb_path,
    upstream_model_type=upstream_model_type,
    frame_pooling_type=frame_pooling_type,
    frame_pooling_param=frame_pooling_param,
    transformer_layer_array=transformer_layer_array,
    device=device
)

train_data, val_data, test_data = loader.load_embedding(
    task_type=task_type,
    subset_percentage=subset_percentage
)


# ----------------------------
# Build downstream model
# ----------------------------
# DownstreamMultiTaskModel should:
# - take embedding inputs from selected layers
# - apply layer pooling across layers (weighted / gated / sap / etc.)
# - pass through shared layers and task heads
model = DownstreamMultiTaskModel(
    upstream_model_type=upstream_model_type,
    task_type=task_type,
    embedding_dim_shared1=embedding_dim_shared1,
    embedding_dim_shared2=embedding_dim_shared2,
    layer_pooling_type=layer_pooling_type,
    layer_pooling_param=layer_pooling_param,
    dropout_prob_shared1=dropout_prob_shared1,
    dropout_prob_shared2=dropout_prob_shared2
)

model.to(device)


# ----------------------------
# Train model
# ----------------------------
# Trainer is responsible for:
# - creating dataloaders
# - running epochs
# - saving checkpoints
# - tracking "best" and "opt" checkpoints via CheckpointManager
trainer = MultiTasksModelTrainer(
    model=model,
    device=device,
    task_type=task_type,
    training_cfg=training_cfg,
    results_root=results_root,
    checkpoints_root=checkpoints_root,
    training_data=train_data,
    validation_data=val_data,
    ignore_index=ignore_index
)

trainer.train()


# ----------------------------
# Evaluate checkpoints
# ----------------------------
# The evaluator resolves checkpoint paths using `checkpoint_tag`.
#
# - "opt"   : loads the checkpoint file whose name contains "opt"
#             expected to be the checkpoint that maximizes the average of per task
#             validation accuracies during training (applicable only for MTL)
#
# - "best"  : loads the checkpoint file whose name contains "best"
#             expected to be the checkpoint that maximizes the overall validation
#             accuracy during training
#
# - "epoch" : loads the final epoch checkpoint by default
#
# - "epochX": loads the checkpoint corresponding to epoch index X (for example "epoch10")
#             The checkpoint file must exist in the checkpoint directory created during training.
#
# NOTE:
# The evaluator loads checkpoint weights into the provided model instance.
# Re using `model=model` is correct because the evaluator overwrites model weights for each checkpoint.

opt_evaluator = MultiTasksModelEvaluator(
    model=model,
    device=device,
    task_type=task_type,
    evaluation_cfg=evaluation_cfg,
    results_root=results_root,
    dataset=test_data,
    checkpoints_root=checkpoints_root,
    checkpoint_tag="opt",
    ignore_index=ignore_index
)
opt_evaluator.run(save_metrics=True, save_predictions=True)

best_evaluator = MultiTasksModelEvaluator(
    model=model,
    device=device,
    task_type=task_type,
    evaluation_cfg=evaluation_cfg,
    results_root=results_root,
    dataset=test_data,
    checkpoints_root=checkpoints_root,
    checkpoint_tag="best",
    ignore_index=ignore_index
)
best_evaluator.run(save_metrics=True, save_predictions=True)

epoch_evaluator = MultiTasksModelEvaluator(
    model=model,
    device=device,
    task_type=task_type,
    evaluation_cfg=evaluation_cfg,
    results_root=results_root,
    dataset=test_data,
    checkpoints_root=checkpoints_root,
    checkpoint_tag="epoch",
    ignore_index=ignore_index
)
epoch_evaluator.run(save_metrics=True, save_predictions=True)
