"""
Main entry script for upstream embedding extraction.

This script:
1. Loads configuration from YAML
2. Loads the upstream SSL model (e.g., WavLM)
3. Builds dataset-specific embedding objects
4. Extracts embeddings
5. Saves embeddings + metadata to disk

This file is intentionally kept as an orchestration layer.
All heavy logic lives in utils/, dataset/, and model/.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import os
import argparse
import torch
import sys
from pathlib import Path

# ----------------------------
# Project imports
# ----------------------------
from dataset.build_embedding import BuildEmbedding
from model.load_upstream import load_upstream_model
from utils.constant_mapping import LabelKeywordMapping
from utils.load_config import load_config
from utils.setup_logging import setup_logging
from utils.setup_device import set_device
from utils.embedding_io import process_and_write_data


# ----------------------------
# NOTE FOR NOTEBOOK USERS
# ----------------------------
# Uncomment this block if running inside Jupyter / IPython
# sys.argv = [
#     "notebook",
#     "--dataset_name", "speechcommand",
#     "--config", "configs/extract_embedding.yml",
#     "--device_index", "0"
# ]


# ----------------------------
# Argument parsing (CLI only)
# ----------------------------
parser = argparse.ArgumentParser(
    description="Extract embeddings using an upstream SSL model"
)

parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help=(
        "Dataset ID to use. "
        "Options: "
        "speechcommand (Google Speech Commands), "
        "voxceleb (VoxCeleb speaker dataset), "
        "iemocap (IEMOCAP emotion dataset), "
        "fluentspeechcommand (Fluent Speech Commands intent dataset)."
    )
)

parser.add_argument(
    "--config",
    type=str,
    default="configs/extract_embedding.yml",
    required=True,
    help="Path to YAML configuration file"
)

parser.add_argument(
    "--device_index",
    type=int,
    default=None,
    help="GPU index to use"
)

args = parser.parse_args()


# ----------------------------
# Load configuration
# ----------------------------
cfg = load_config(args.config)

# Paths
root_data_path = cfg["paths"]["root_data_path"]
root_emb_path = cfg["paths"]["root_emb_path"]
checkpoint_path = cfg["paths"]["upstream_checkpoint_path"]

# Dataset
dataset_name = args.dataset_name
dataset_limit = cfg["dataset"]["limit"]      

# Upstream model
upstream_model_type = cfg["upstream"]["model_type"]

# Pooling
frame_pooling_type = cfg["pooling"]["frame_pooling_type"]
frame_pooling_param = cfg["pooling"]["frame_pooling_param"]

# Device
device_type = cfg["device"]["type"]
device_index = (
    args.device_index
    if args.device_index is not None
    else cfg["device"]["index"]
)

# Logging
log_level = cfg["log_level"]


# ----------------------------
# Setup logging
# ----------------------------
setup_logging(log_level=log_level)


# ----------------------------
# Setup device (CPU / GPU)
# ----------------------------
device = set_device(
    device_type=device_type,
    device_index=device_index
)


# ----------------------------
# Load upstream SSL model
# ----------------------------
upstream_model = load_upstream_model(
    upstream_model_type=upstream_model_type,
    checkpoint_path=checkpoint_path,
    device=device
)


# ----------------------------
# Build dataset embedding objects
# ----------------------------
# This step prepares dataset wrappers that:
# - load audio
# - extract SSL representations
# - apply frame pooling
data_embedding = BuildEmbedding(
    upstream_model=upstream_model,
    frame_pooling_type=frame_pooling_type,
    frame_pooling_param=frame_pooling_param,
    device=device
)

training_data, validating_data, testing_data = data_embedding.build_embedding(
    dataset_name=dataset_name,
    root_data_path=root_data_path
)


# ----------------------------
# Write embeddings to disk
# ----------------------------
# All splits are written to the SAME CSV
# append=False → create new CSV
# append=True  → add to existing CSV

process_and_write_data(
    data=training_data,
    root_emb_path=root_emb_path,
    upstream_model_type=upstream_model_type,
    frame_pooling_type=frame_pooling_type,
    frame_pooling_param=frame_pooling_param,
    dataset_name=dataset_name,
    limit=dataset_limit,
    append=False,
)

process_and_write_data(
    data=validating_data,
    root_emb_path=root_emb_path,
    upstream_model_type=upstream_model_type,
    frame_pooling_type=frame_pooling_type,
    frame_pooling_param=frame_pooling_param,
    dataset_name=dataset_name,
    limit=dataset_limit,
    append=True,
)

process_and_write_data(
    data=testing_data,
    root_emb_path=root_emb_path,
    upstream_model_type=upstream_model_type,
    frame_pooling_type=frame_pooling_type,
    frame_pooling_param=frame_pooling_param,
    dataset_name=dataset_name,
    limit=dataset_limit,
    append=True,
)

