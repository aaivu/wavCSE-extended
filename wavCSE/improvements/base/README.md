# Base (plain wavCSE, MLflow-tracked)

Not an owner's MTL architecture -- this runs the original, unmodified
`DownstreamMultiTaskModel` (same model `run_improvements.py --model original`
would use) so its results can serve as the baseline every architecture in
`taskrelation/`, `lowrank/`, `clustering/`, `decomposition/` is compared
against. It exists to add MLflow experiment tracking (backed by a
DagsHub-hosted tracking server) without touching `downstream/`.

- `run_base.py` -- orchestration script: loads embeddings, builds the plain
  `DownstreamMultiTaskModel`, trains and evaluates it via the unmodified
  `downstream/trainer` and `downstream/evaluator`, and logs params/metrics/
  artifacts to MLflow using `improvements/mlflow_utils.py`.
- `configs/base_config.yml` -- full copy of `downstream/configs/build_model.yml`'s
  schema plus an `mlflow:` block (`tracking_uri`, `experiment_name`).

## Running

```bash
cd wavCSE/wavCSE/improvements/base
python run_base.py --task_type ks_si_er --config configs/base_config.yml --device_index 0
```

Can be run from any working directory -- all paths (downstream/, improvements/,
this folder's own config) are resolved relative to `run_base.py`'s own location,
not the working directory.

## MLflow / DagsHub setup

1. Copy `.env.example` (at the `wavCSE/wavCSE/` repo root) to `.env` and fill in
   your DagsHub username + access token. `.env` is gitignored -- never commit it.
2. `base_config.yml`'s `mlflow.tracking_uri` points at
   `https://dagshub.com/Ke-vin-S/wavCSE.mlflow`. Change it if you're tracking
   against a different DagsHub repo.
3. Each run appears under the `wavcse-baseline` experiment on DagsHub's
   Experiments tab, with logged hyperparameters, per-epoch train/val loss and
   accuracy curves (overall + per task), final test metrics for the `opt`/
   `best`/`epoch` checkpoints, and artifacts (plots, checkpoints, prediction
   CSVs) pulled from the same `results_base/`/`checkpoints_base/` directories
   this script writes locally regardless of MLflow.
