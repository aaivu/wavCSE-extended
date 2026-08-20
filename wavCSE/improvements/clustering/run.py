"""Train and evaluate the three-task NCMTL downstream model.

Usage from this directory:
    python run.py --task_type ks_si_er --config configs/ncmtl_config.yml --device_index 0

All repository paths are resolved relative to this file, so the runner can also
be invoked from another working directory.
"""

import argparse
import logging
import os
import sys
from contextlib import nullcontext
from datetime import datetime

from dotenv import load_dotenv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPROVEMENTS_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_IMPROVEMENTS_DIR)
_DOWNSTREAM_DIR = os.path.join(_REPO_ROOT, "downstream")

# Avoid exposing clustering/utils as the top-level `utils` package. The legacy
# downstream code expects its own namespace package at downstream/utils.
sys.path = [
    path
    for path in sys.path
    if os.path.abspath(path or os.getcwd()) != _THIS_DIR
]
sys.path.insert(0, _IMPROVEMENTS_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _DOWNSTREAM_DIR)

from dataset.load_embedding import LoadEmbedding
from evaluator.evaluator_model import MultiTasksModelEvaluator
from improvements.clustering.models.ncmtl_model import DownstreamMultiTaskModelNCMTL
from improvements.clustering.trainers.ncmtl_trainer import MultiTasksModelTrainerNCMTL
from improvements.tensorboard_utils import tensorboard_trainer
from utils.load_config import load_config
from utils.parse_transformer_layers import parse_transformer_layers
from utils.setup_device import set_device
from utils.setup_logging import setup_logging


def _mlflow_run_context(cfg, task_type):
    """Start remote tracking when available, otherwise keep the local run usable."""
    try:
        import mlflow
        import mlflow_utils

        mlflow_utils.setup_mlflow(cfg)
        run_name = f"ncmtl_{task_type}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        return mlflow.start_run(run_name=run_name)
    except Exception as error:
        logging.warning(
            "MLflow/DagsHub is unavailable; continuing with local artifacts: %s",
            error,
        )
        return nullcontext(None)


def _log_initial_mlflow_state(cfg, task_type):
    try:
        import mlflow
        import mlflow_utils

        mlflow_utils.log_config_params(cfg)
        mlflow.log_param("task_type", task_type)
    except Exception as error:
        logging.warning("Could not log initial MLflow parameters: %s", error)


def _log_final_mlflow_state(trainer, model):
    try:
        import mlflow

        for tag in ("best", "opt"):
            path = trainer.model_checkpoint_path.replace(".pth", f"_{tag}.pth")
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="checkpoints")

        state = model.get_cluster_state()
        for task, cluster in zip(trainer.task_array, state["assignments"]):
            mlflow.log_param(f"final_cluster_{task}", cluster)
        mlflow.log_param("clusters_frozen", state["frozen"])
        mlflow.log_param(
            "cluster_frozen_epoch",
            trainer.frozen_epoch if trainer.frozen_epoch is not None else "not_frozen",
        )
        mlflow.log_artifacts(trainer.results_dir, artifact_path="results")
    except Exception as error:
        logging.warning("Could not log final MLflow artifacts: %s", error)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the three-task wavCSE-NCMTL model"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="ks_si_er",
        help="NCMTL v1 supports only ks_si_er",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_THIS_DIR, "configs", "ncmtl_config.yml"),
        help="Path to the NCMTL YAML configuration",
    )
    parser.add_argument(
        "--device_index",
        type=int,
        default=None,
        help="GPU index to use (overrides the configuration)",
    )
    args = parser.parse_args()

    if args.task_type != "ks_si_er":
        parser.error("NCMTL v1 supports only --task_type ks_si_er")

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    cfg = load_config(args.config)
    setup_logging(log_level=cfg["log_level"])

    device_index = (
        args.device_index
        if args.device_index is not None
        else cfg["device"]["index"]
    )
    device = set_device(
        device_type=cfg["device"]["type"], device_index=device_index
    )

    transformer_layer_array = parse_transformer_layers(
        transformer_layers=cfg["upstream"]["selected_transformer_layers"],
        upstream_model_type=cfg["upstream"]["model_type"],
    )
    layer_pooling_type = cfg["pooling"]["layer_pooling_type"]
    layer_pooling_param = (
        cfg["pooling"]["layer_pooling_param"]
        if layer_pooling_type not in ("weighted", "gated")
        else len(transformer_layer_array)
    )

    loader = LoadEmbedding(
        root_data_path=cfg["paths"]["root_data_path"],
        root_emb_path=cfg["paths"]["root_emb_path"],
        upstream_model_type=cfg["upstream"]["model_type"],
        frame_pooling_type=cfg["pooling"]["frame_pooling_type"],
        frame_pooling_param=cfg["pooling"]["frame_pooling_param"],
        transformer_layer_array=transformer_layer_array,
        device=device,
    )
    train_data, val_data, test_data = loader.load_embedding(
        task_type=args.task_type,
        subset_percentage=cfg["dataset"]["subset_percentage"],
    )

    with _mlflow_run_context(cfg, args.task_type) as active_run:
        if active_run is not None:
            _log_initial_mlflow_state(cfg, args.task_type)

        model = DownstreamMultiTaskModelNCMTL(
            upstream_model_type=cfg["upstream"]["model_type"],
            task_type=args.task_type,
            embedding_dim_shared1=cfg["model"]["embedding_dim_shared1"],
            embedding_dim_shared2=cfg["model"]["embedding_dim_shared2"],
            layer_pooling_type=layer_pooling_type,
            layer_pooling_param=layer_pooling_param,
            dropout_prob_shared1=cfg["model"]["dropout_prob_shared1"],
            dropout_prob_shared2=cfg["model"]["dropout_prob_shared2"],
        ).to(device)

        trainer_class = tensorboard_trainer(MultiTasksModelTrainerNCMTL)
        trainer = trainer_class(
            model=model,
            device=device,
            task_type=args.task_type,
            training_cfg=cfg["training"],
            results_root=cfg["paths"]["results_root"],
            checkpoints_root=cfg["paths"]["checkpoints_root"],
            training_data=train_data,
            validation_data=val_data,
            ignore_index=cfg["dataset"]["ignore_index"],
            ncmtl_cfg=cfg["ncmtl"],
            tensorboard_cfg=cfg.get("tensorboard", {}),
        )
        trainer.train()

        run_id = os.path.basename(trainer.results_dir)
        checkpoint_run_id = os.path.basename(trainer.ckpt_dir)
        for tag in ("opt", "best", "epoch"):
            evaluator = MultiTasksModelEvaluator(
                model=model,
                device=device,
                task_type=args.task_type,
                evaluation_cfg=cfg["evaluation"],
                results_root=cfg["paths"]["results_root"],
                dataset=test_data,
                checkpoints_root=cfg["paths"]["checkpoints_root"],
                checkpoint_tag=tag,
                ignore_index=cfg["dataset"]["ignore_index"],
                results_run_id=run_id,
                checkpoint_run_id=checkpoint_run_id,
            )
            stats = evaluator.write_metrics()
            evaluator.write_predictions_csv()
            if active_run is not None:
                try:
                    import mlflow_utils

                    mlflow_utils.log_eval_stats(stats, tag, trainer.task_array)
                except Exception as error:
                    logging.warning("Could not log MLflow evaluation metrics: %s", error)

        if active_run is not None:
            _log_final_mlflow_state(trainer, model)


if __name__ == "__main__":
    main()
