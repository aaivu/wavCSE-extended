# Network Clustering for Multi-Task Learning (NCMTL)

Owner: Induwara
MTL branch: Task Clustering

This folder implements the first NCMTL milestone for exactly three tasks:
keyword spotting (`ks`), speaker identification (`si`), and emotion
recognition (`er`). The supported task string is `ks_si_er`; intent
classification is not supported by NCMTL v1.

## Architecture

The baseline FC1, layer pooling, and FC2 path remains shared. After the
2000-dimensional FC2 representation, every task has a cluster candidate layer
with shape `2000 -> 2000` and no bias. Its dimension is derived automatically
from the actual FC2 output dimension, so NCMTL introduces clustering without an
additional representation bottleneck. K-Means++ groups only the three candidate
weight matrices. Tasks in a cluster are projected to their exact PyTorch mean
after optimizer updates, while the final classifiers remain task-specific.

`get_all_embeddings()` continues to return the common FC2 representation before
the candidate layers.

## Layout

```text
clustering/
├── run.py
├── configs/ncmtl_config.yml
├── models/ncmtl_model.py
├── trainers/ncmtl_trainer.py
├── utils/ncmtl_clustering.py
└── tests/test_ncmtl.py
```

## Key configuration

- `num_clusters`: K for K-Means++; initially 2 and valid from 1 through 3.
- `alpha`: cluster-loss coefficient; initially 0.001.
- `cluster_every_n_batches`: training clustering interval; initially every batch.
- `warmup_epochs`: independent candidate training before clustering; initially 0.
- `stability_patience`: unchanged partitions needed before freezing.
- `max_recluster_epochs`: last epoch during which K-Means++ may run.

These values are initial experiment settings, not claimed optimal values.

## Run

From any working directory:

```bash
cd improvements/clustering
conda activate tc
python run.py \
  --task_type ks_si_er \
  --config configs/ncmtl_config.yml \
  --device_index 0
```

The runner is independent of `improvements/run_improvements.py`. Training and
evaluation use `results_ncmtl/` and `checkpoints_ncmtl/`. Each
result run contains `cluster_history.csv` and `cluster_summary.json`, alongside
the inherited metrics, plots, and prediction files. When configured, the run is
also logged to MLflow/DagsHub; unavailable remote credentials do not prevent
local execution.

K-Means++ runs only on training batches. Validation and evaluation never
recluster. Once the assignment freezes, K-Means++ stops, but mean projection
continues after every optimizer update so cluster members remain shared.

- To run:

```bash
  nohup python run.py \
    --task_type ks_si_er \
    --config configs/ncmtl_config.yml \
    --device_index 0 \
    > /tmp/run_ncmtl_ks_si_er.log 2>&1 < /dev/null &

  disown
```
