# wavCSE

wavCSE is a modular speech representation learning framework designed for systematic and reproducible research in self supervised speech representation learning and downstream multi task modeling.

The framework enforces a clear separation between upstream embedding extraction and downstream model training and evaluation.

---

## Overview

The wavCSE framework follows a two stage pipeline executed sequentially.

### Stage 1: Upstream Embedding Extraction  
Pretrained self supervised learning models are used to extract fixed dimensional speech representations from raw audio and store them on disk.

### Stage 2: Downstream Model Training and Evaluation  
Downstream multi task neural networks are trained and evaluated using the extracted embeddings.

**Important**  
Stage 1 must be executed from the `upstream/` directory.  
Stage 2 must be executed from the `downstream/` directory.  
Upstream embedding extraction must be completed before downstream training.

---

## Project Structure

```text
wavCSE/
├── upstream/          # Upstream SSL based embedding extraction
├── downstream/        # Downstream multi task training and evaluation
├── environment.yml    # Conda environment definition
└── README.md          # Project documentation
```

---

### Environment setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate opencv
```

Core libraries (already included in environment.yml):
- python 3.9
- torch 2.7.1
- torchaudio 2.7.1

---

## Stage 1: Upstream Embedding Extraction

This stage loads a pretrained upstream model and converts raw audio signals into fixed dimensional speech embeddings that are serialized to disk.

### Execution

Navigate to the `upstream/` directory before running the extraction script.

```bash
cd upstream
python main.py \
  --dataset_name speechcommand \
  --config configs/extract_embedding.yml \
  --device_index 0
```
  
### Arguments

| Argument | Description |
| --- | --- |
| `--dataset_name` | Dataset identifier. Must be one of the supported dataset codes listed below |
| `--config` | Path to upstream YAML configuration file |
| `--device_index` | GPU index to use (optional, overrides configuration file) |

---

## Supported Datasets

The following datasets are currently supported by wavCSE.  
Only the dataset codes listed below are valid values for the `--dataset_name` argument.

| Dataset Code | Dataset Name |
| --- | --- |
| `speechcommand` | Google Speech Commands v1 |
| `voxceleb` | VoxCeleb1 |
| `iemocap` | IEMOCAP |
| `fluentspeechcommand` | Fluent Speech Commands |

**Note**  
For each dataset, the folder name must exactly match the dataset code shown above.  
The internal directory structure of the dataset must remain unchanged.

### Output

- Serialized embedding files (`.pt`)
- A CSV file containing:
  - Audio path
  - Label
  - Embedding file size

---

## Stage 2: Downstream Model Training and Evaluation

This stage loads the extracted embeddings, constructs a downstream multi task model, trains the model, and evaluates multiple checkpoint variants.

### Execution

Navigate to the `downstream/` directory before running the training and evaluation script.

```bash
cd downstream
python main.py \
  --task_type ks_si_er_ic \
  --config configs/build_model.yml \
  --device_index 0
```

### Arguments

| Argument | Description |
| --- | --- |
| `--task_type` | Task combination string specifying the downstream task configuration |
| `--config` | Path to downstream YAML configuration file |
| `--device_index` | GPU index to use (optional, overrides configuration file) |

### Supported Tasks

| Code | Task                   |
|------|------------------------|
| `ks` | Keyword Spotting       |
| `si` | Speaker Identification |
| `er` | Emotion Recognition    |
| `ic` | Intent Classification  |

#### Task Type Specification

The `task_type` argument defines the set of downstream tasks to be trained jointly.

**Single task configurations**
- `ks`  : Keyword Spotting  
- `si`  : Speaker Identification  
- `er`  : Emotion Recognition  
- `ic`  : Intent Classification  

**Multi task configurations**  
Multiple tasks are specified by concatenating task codes using underscores.

**Two task combinations**
- `ks_si`
- `ks_er`
- `si_er`
- `er_ic`

**Three task combinations**
- `ks_si_er`
- `ks_er_ic`
- `si_er_ic`

**Four task combination**
- `ks_si_er_ic`

All specified tasks are trained jointly using a shared downstream model.

---

## Checkpoints and Evaluation

The downstream pipeline supports evaluation of multiple checkpoint types.

| Checkpoint | Description |
| --- | --- |
| `best` | Checkpoint that achieves the best overall validation accuracy during training |
| `opt` | Checkpoint that achieves the best average task accuracy during training (applicable only for multi task learning) |
| `epochX` | Checkpoint corresponding to epoch index `X`. If `X` is not specified (i.e., `epoch`), the final epoch checkpoint is used by default |

For the `epochX` option, the corresponding checkpoint must exist in the checkpoint directory created during training.
For example, specifying `epoch10` loads the checkpoint saved at the tenth training epoch.

### Evaluation Outputs

- Metric summaries  
- Per-task accuracy  
- Prediction CSV files  

---

## Pretrained Models and Checkpoints

### Upstream model checkpoints

wavCSE uses pretrained self supervised models for upstream embedding extraction.

The latest official **WavLM Large** checkpoint is automatically downloaded when running the upstream embedding extraction stage. No manual download is required.

### Downstream pretrained checkpoints

Pretrained downstream multi task models are available for selected task configurations.

For the task combination:

- **ks_si_er** (Keyword Spotting + Speaker Identification + Emotion Recognition)

Pretrained checkpoints can be downloaded from:

https://drive.google.com/drive/folders/1kRlOutpEg9Ii7BR4QbntdSr01ZmIjdgo?usp=drive_link

The folder contains the following checkpoint files:

- `wavlm_large_ks_si_er_best.pth`  
  Best validation checkpoint

- `wavlm_large_ks_si_er_epoch100.pth`  
  Checkpoint saved at epoch 100

- `wavlm_large_ks_si_er_opt.pth`  
  Best average task performance checkpoint (multi task optimization)

#### Usage

The checkpoint can then be evaluated using the corresponding checkpoint option (`best`, `epoch100`, or `opt`) in the downstream evaluation stage.

---

## Configuration

All experimental settings are controlled via YAML configuration files, including:

- Upstream model selection  
- Transformer layer selection  
- Pooling strategies  
- Training hyperparameters  
- Evaluation behavior  

This design enables fully reproducible experiments without modifying code.

---

## Usage Notes

- Stage 1 must be executed from the `upstream/` directory  
- Stage 2 must be executed from the `downstream/` directory  
- Upstream embedding extraction must be completed before downstream training  
- GPU usage can be controlled through configuration files or command line overrides  

---

## Extensibility

The wavCSE framework is designed to support future research extensions, including:

- Additional upstream models such as HuBERT and wav2vec 2.0  
- New pooling and aggregation strategies  
- Additional downstream tasks  
- Analysis and visualization modules  

---

## Contact

For questions, or feedback related to wavCSE, please contact:

- **Braveenan Sritharan**  
  braveenans.22@cse.mrt.ac.lk

- **Uthayasanker Thayasivam**  
  rtuthaya@cse.mrt.ac.lk

---

## License

This project is intended for research and academic use.  