"""
Run script for wavCSE architectural improvements.

This script benchmarks three improved wavCSE architectures against the original:

1. wavCSE-GBC: Global Bias Coupling (Zhang et al. 2010)
   - Adds a shared global bias that couples all task heads
   - Lightest change, most similar to original

2. wavCSE-TSM: Task Structure Matrix (Ciliberto et al. 2015)
   - Replaces independent heads with latent classifier basis + structure matrix A
   - Alternating minimization between model and structure
   - l1 sparsity on task relations

3. wavCSE-PMR: Precision Matrix Regularization (Goncalves et al. 2016)
   - Adds learnable sparse precision matrix Omega over task parameters
   - Gaussian graphical model interpretation
   - Trace + log-det regularization

Original wavCSE contributions (layer selection, frame/layer pooling) are PRESERVED.

Usage:
    python run_improvements.py --model [gbc|tsm|pmr|all] --task_type ks_si_er
"""

import os
import sys
import argparse
import logging
import yaml
from datetime import datetime

import torch

# Add downstream path for imports
DOWNSTREAM_PATH = os.path.join(
    os.path.dirname(__file__), '../downstream'
)
sys.path.insert(0, DOWNSTREAM_PATH)

from dataset.load_embedding import LoadEmbedding
from utils.load_config import load_config
from utils.setup_device import set_device
from utils.setup_logging import setup_logging
from utils.parse_transformer_layers import parse_transformer_layers


def build_model(model_type: str, cfg: dict, task_type: str, layer_pooling_param):
    """Factory for building the appropriate model variant."""
    model_cfg = cfg.get("model", {})

    common_args = dict(
        upstream_model_type=cfg["upstream"]["model_type"],
        task_type=task_type,
        embedding_dim_shared1=model_cfg.get("embedding_dim_shared1", 512),
        embedding_dim_shared2=model_cfg.get("embedding_dim_shared2", 2000),
        layer_pooling_type=cfg["pooling"]["layer_pooling_type"],
        dropout_prob_shared1=model_cfg.get("dropout_prob_shared1", 0.4),
        dropout_prob_shared2=model_cfg.get("dropout_prob_shared2", 0.6),
        layer_pooling_param=layer_pooling_param,
    )

    if model_type == "gbc":
        from improvements.taskrelation.models.gbc_model import DownstreamMultiTaskModelGBC
        return DownstreamMultiTaskModelGBC(
            **common_args,
            gbc_global_dim=model_cfg.get("gbc_global_dim", 64),
        )
    elif model_type == "tsm":
        from improvements.taskrelation.models.tsm_model import DownstreamMultiTaskModelTSM
        return DownstreamMultiTaskModelTSM(
            **common_args,
            num_latent_classifiers=model_cfg.get("num_latent_classifiers", 8),
            structure_sparsity_lambda=model_cfg.get("structure_sparsity_lambda", 0.01),
        )
    elif model_type == "pmr":
        from improvements.taskrelation.models.pmr_model import DownstreamMultiTaskModelPMR
        return DownstreamMultiTaskModelPMR(
            **common_args,
            pmr_lambda=model_cfg.get("pmr_lambda", 0.01),
            pmr_gamma=model_cfg.get("pmr_gamma", 0.1),
        )
    elif model_type == "original":
        from model.downstream_model import DownstreamMultiTaskModel
        return DownstreamMultiTaskModel(**common_args)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def build_trainer(model_type: str, model, device, task_type, cfg, training_data,
                  validation_data, ignore_index):
    """Factory for building the appropriate trainer."""
    training_cfg = cfg["training"]

    if model_type == "tsm":
        from improvements.taskrelation.trainers.tsm_trainer import MultiTasksModelTrainerTSM
        tsm_cfg = cfg.get("tsm", {})
        return MultiTasksModelTrainerTSM(
            model=model,
            device=device,
            task_type=task_type,
            training_cfg=training_cfg,
            results_root=cfg["paths"]["results_root"],
            checkpoints_root=cfg["paths"]["checkpoints_root"],
            training_data=training_data,
            validation_data=validation_data,
            ignore_index=ignore_index,
            alternate_frequency=tsm_cfg.get("alternate_frequency", 5),
            structure_epochs=tsm_cfg.get("structure_epochs", 1),
            structure_lr=tsm_cfg.get("structure_lr", 0.001),
        )
    elif model_type == "pmr":
        from improvements.taskrelation.trainers.pmr_trainer import MultiTasksModelTrainerPMR
        pmr_cfg = cfg.get("pmr", {})
        return MultiTasksModelTrainerPMR(
            model=model,
            device=device,
            task_type=task_type,
            training_cfg=training_cfg,
            results_root=cfg["paths"]["results_root"],
            checkpoints_root=cfg["paths"]["checkpoints_root"],
            training_data=training_data,
            validation_data=validation_data,
            ignore_index=ignore_index,
            pmr_warmup_epochs=pmr_cfg.get("warmup_epochs", 3),
        )
    else:
        # GBC and original use the standard trainer
        from trainer.trainer_model import MultiTasksModelTrainer
        return MultiTasksModelTrainer(
            model=model,
            device=device,
            task_type=task_type,
            training_cfg=training_cfg,
            results_root=cfg["paths"]["results_root"],
            checkpoints_root=cfg["paths"]["checkpoints_root"],
            training_data=training_data,
            validation_data=validation_data,
            ignore_index=ignore_index,
        )


def run_single_model(model_type: str, task_type: str, config_path: str,
                     device_index: int):
    """Run a single model variant end-to-end."""
    print(f"\n{'='*60}")
    print(f"  Running wavCSE-{model_type.upper()}")
    print(f"  Task: {task_type}")
    print(f"{'='*60}\n")

    cfg = load_config(config_path)
    setup_logging(cfg.get("log_level", "INFO"))

    device = set_device(
        cfg.get("device", {}).get("type", "cuda"),
        device_index if device_index is not None else cfg.get("device", {}).get("index", 0)
    )

    # Parse transformer layers
    transformer_layer_array = parse_transformer_layers(
        cfg["upstream"]["selected_transformer_layers"],
        cfg["upstream"]["model_type"]
    )
    logging.info(f"Selected transformer layers: {transformer_layer_array}")

    # Load embeddings
    loader = LoadEmbedding(
        root_emb_path=os.path.expanduser(cfg["paths"]["root_emb_path"]),
        root_data_path=os.path.expanduser(cfg["paths"]["root_data_path"]),
        upstream_model_type=cfg["upstream"]["model_type"],
        frame_pooling_type=cfg["pooling"]["frame_pooling_type"],
        frame_pooling_param=cfg["pooling"].get("frame_pooling_param"),
        selected_transformer_layers=transformer_layer_array,
        device=device,
    )

    subset_pct = cfg["dataset"].get("subset_percentage", 100)
    train_data, val_data, test_data = loader.load_embedding(
        task_type, subset_percentage=subset_pct
    )

    # Determine layer pooling param
    layer_pooling_type = cfg["pooling"]["layer_pooling_type"]
    layer_pooling_param = cfg["pooling"].get("layer_pooling_param")
    if layer_pooling_type in ["weighted", "gated"]:
        layer_pooling_param = len(transformer_layer_array)

    # Build model
    model = build_model(model_type, cfg, task_type, layer_pooling_param)
    model.to(device)

    # Build trainer
    ignore_index = cfg["dataset"].get("ignore_index", -1)
    trainer = build_trainer(
        model_type, model, device, task_type, cfg,
        train_data, val_data, ignore_index
    )

    # Train
    trainer.train()

    # Evaluate
    from evaluator.evaluator_model import MultiTasksModelEvaluator
    for tag in ["opt", "best", "epoch"]:
        try:
            evaluator = MultiTasksModelEvaluator(
                model=model,
                device=device,
                task_type=task_type,
                checkpoints_root=cfg["paths"]["checkpoints_root"],
                results_root=cfg["paths"]["results_root"],
                eval_cfg=cfg.get("evaluation", {}),
                test_data=test_data,
                checkpoint_tag=tag,
                ignore_index=ignore_index,
            )
            evaluator.run()
        except Exception as e:
            logging.warning(f"Evaluation with tag='{tag}' failed: {e}")

    print(f"\n  wavCSE-{model_type.upper()} completed.\n")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Run wavCSE architectural improvements"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["gbc", "tsm", "pmr", "original", "all"],
        help="Which model variant to run"
    )
    parser.add_argument(
        "--task_type", type=str, default="ks_si_er",
        help="Task type (e.g., ks_si_er)"
    )
    parser.add_argument(
        "--device_index", type=int, default=None,
        help="GPU device index"
    )
    args = parser.parse_args()

    # "gbc"/"tsm"/"pmr" are all task-relation-learning variants (Kevin's branch).
    # When other members add their own architecture folders (lowrank/, clustering/,
    # decomposition/), extend this lookup accordingly.
    config_dir = os.path.join(os.path.dirname(__file__), "taskrelation", "configs")

    if args.model == "all":
        models_to_run = ["gbc", "tsm", "pmr"]
    else:
        models_to_run = [args.model]

    results = {}
    for mtype in models_to_run:
        config_path = os.path.join(config_dir, f"{mtype}_config.yml")
        if not os.path.exists(config_path):
            print(f"Warning: No config found for {mtype} at {config_path}")
            continue
        try:
            results[mtype] = run_single_model(
                mtype, args.task_type, config_path, args.device_index
            )
        except Exception as e:
            print(f"Error running {mtype}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("  All model runs completed.")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
