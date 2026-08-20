"""
Run script for the plain wavCSE downstream model (no new MTL architecture),
with MLflow experiment tracking (backed by a DagsHub-hosted tracking server).

This deliberately reuses downstream/dataset/load_embedding.py,
downstream/model/downstream_model.py, downstream/trainer/trainer_model.py,
and downstream/evaluator/evaluator_model.py UNMODIFIED -- downstream/ is not
touched by this script. MLflow logging only reads their public
attributes/return values after the fact (see improvements/mlflow_utils.py).

Usage:
    python run_base.py --task_type ks_si_er --config configs/base_config.yml [--device_index 0]

Can be invoked from any working directory: all cross-package paths are
resolved relative to this file's own location, not the working directory.
"""

import os
import sys
import argparse
from datetime import datetime

from dotenv import load_dotenv

# ----------------------------
# Path setup (robust to cwd)
# ----------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPROVEMENTS_DIR = os.path.dirname(_THIS_DIR)              # .../improvements
_REPO_ROOT = os.path.dirname(_IMPROVEMENTS_DIR)              # .../wavCSE (git repo root)
_DOWNSTREAM_DIR = os.path.join(_REPO_ROOT, "downstream")

sys.path.insert(0, _DOWNSTREAM_DIR)     # exposes dataset/, model/, trainer/, evaluator/, utils/ as top-level
sys.path.insert(0, _IMPROVEMENTS_DIR)   # exposes mlflow_utils as top-level

import mlflow
import mlflow_utils

from dataset.load_embedding import LoadEmbedding
from model.downstream_model import DownstreamMultiTaskModel
from trainer.trainer_model import MultiTasksModelTrainer
from evaluator.evaluator_model import MultiTasksModelEvaluator
from utils.load_config import load_config
from utils.setup_device import set_device
from utils.setup_logging import setup_logging
from utils.parse_transformer_layers import parse_transformer_layers


class _LiveMlflowTrainer(MultiTasksModelTrainer):
    """MultiTasksModelTrainer, with one hook added: after each epoch's train/val
    stats are computed, push them to MLflow immediately (instead of waiting for
    the whole run to finish). Overrides only _epoch_report_line -- the smallest
    point in trainer_model.py's train() loop that already receives the full
    structured EpochStats per phase, called twice per epoch (once for "train",
    once for "val"). Everything else is inherited unmodified from
    downstream/trainer/trainer_model.py.

    Coupling risk: relies on _epoch_report_line's private signature
    (epoch, phase, stats) staying stable in trainer_model.py.
    """

    def _epoch_report_line(self, epoch, phase, stats):
        learning_rate = None
        if phase == "val":
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
        mlflow_utils.log_epoch_stats(epoch, phase, stats, self.task_array, learning_rate=learning_rate)
        return super()._epoch_report_line(epoch, phase, stats)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the plain wavCSE downstream model, tracked in MLflow"
    )
    parser.add_argument("--task_type", type=str, required=True,
                         help="Task string joined by underscores (e.g., ks_si_er_ic)")
    parser.add_argument("--config", type=str,
                         default=os.path.join(_THIS_DIR, "configs", "base_config.yml"),
                         help="Path to YAML configuration file")
    parser.add_argument("--device_index", type=int, default=None,
                         help="GPU index to use (overrides config file)")
    args = parser.parse_args()

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))

    # ----------------------------
    # Load configuration
    # ----------------------------
    cfg = load_config(args.config)

    log_level = cfg["log_level"]
    task_type = args.task_type

    device_type = cfg["device"]["type"]
    device_index = args.device_index if args.device_index is not None else cfg["device"]["index"]

    root_data_path = cfg["paths"]["root_data_path"]
    root_emb_path = cfg["paths"]["root_emb_path"]
    results_root = cfg["paths"]["results_root"]
    checkpoints_root = cfg["paths"]["checkpoints_root"]

    upstream_model_type = cfg["upstream"]["model_type"]
    selected_transformer_layers = cfg["upstream"]["selected_transformer_layers"]
    transformer_layer_array = parse_transformer_layers(
        transformer_layers=selected_transformer_layers,
        upstream_model_type=upstream_model_type
    )

    subset_percentage = cfg["dataset"]["subset_percentage"]
    ignore_index = cfg["dataset"]["ignore_index"]

    frame_pooling_type = cfg["pooling"]["frame_pooling_type"]
    layer_pooling_type = cfg["pooling"]["layer_pooling_type"]
    frame_pooling_param = cfg["pooling"]["frame_pooling_param"]
    layer_pooling_param = (
        cfg["pooling"]["layer_pooling_param"]
        if layer_pooling_type not in ["weighted", "gated"]
        else len(transformer_layer_array)
    )

    embedding_dim_shared1 = cfg["model"]["embedding_dim_shared1"]
    embedding_dim_shared2 = cfg["model"]["embedding_dim_shared2"]
    dropout_prob_shared1 = cfg["model"]["dropout_prob_shared1"]
    dropout_prob_shared2 = cfg["model"]["dropout_prob_shared2"]

    training_cfg = cfg["training"]
    evaluation_cfg = cfg["evaluation"]

    setup_logging(log_level=log_level)
    device = set_device(device_type=device_type, device_index=device_index)

    # ----------------------------
    # MLflow setup
    # ----------------------------
    mlflow_utils.setup_mlflow(cfg)
    run_name = f"base_{task_type}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow_utils.log_config_params(cfg)
        mlflow.log_param("task_type", task_type)

        # ----------------------------
        # Load embeddings
        # ----------------------------
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
        # Build model
        # ----------------------------
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
        # Train
        # ----------------------------
        trainer = _LiveMlflowTrainer(
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

        for tag in ["best", "opt"]:
            ckpt_path = trainer.model_checkpoint_path.replace(".pth", f"_{tag}.pth")
            if os.path.exists(ckpt_path):
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

        # ----------------------------
        # Evaluate opt / best / epoch checkpoints
        # ----------------------------
        task_array = trainer.task_array
        for tag in ["opt", "best", "epoch"]:
            evaluator = MultiTasksModelEvaluator(
                model=model,
                device=device,
                task_type=task_type,
                evaluation_cfg=evaluation_cfg,
                results_root=results_root,
                dataset=test_data,
                checkpoints_root=checkpoints_root,
                checkpoint_tag=tag,
                ignore_index=ignore_index
            )
            stats = evaluator.write_metrics()
            evaluator.write_predictions_csv()
            mlflow_utils.log_eval_stats(stats, tag, task_array)

        # ----------------------------
        # Log all local results (plots, txt logs, eval metrics, prediction CSVs)
        # ----------------------------
        mlflow.log_artifacts(trainer.results_dir, artifact_path="results")


if __name__ == "__main__":
    main()
