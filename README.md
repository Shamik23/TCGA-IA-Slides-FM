# TCGA Survival Foundation Model

Train a censored survival model for TCGA solid tumors on frozen histopathology
foundation-model embeddings with an attention-MIL Cox head. The repository does
not read whole-slide images or perform tissue detection; it expects pre-tiled
image folders or precomputed feature files.

## Pipeline

1. Fetch TCGA clinical survival labels from the GDC API.
2. Prepare diagnostic slide tiles with an external tiling tool.
3. Cache tile features with a frozen pathology encoder, or convert a public
  UNI2-h `.h5` feature archive.
4. Build a manifest that joins slide features to patient survival labels.
5. Train the attention-MIL Cox head on cached features.
6. Evaluate with the censored concordance index (c-index).

Tiling and feature extraction are kept outside the training loop. Training reads
variable-length feature bags, pads each batch, applies gated attention pooling,
and predicts one Cox risk score per slide. Splits are made by patient to avoid
slide leakage.

## Installation

Pick one of the three options.

### Docker (recommended)

build directly with options:

```bash
# CPU-only, production (default)
docker build -t tcga-survival-fm:latest .

# CPU, development (adds pytest, mypy, bandit, pre-commit)
docker build --build-arg INSTALL_DEV=true -t tcga-survival-fm:latest .

# NVIDIA GPU, production
docker build --build-arg BASE=gpu-base -t tcga-survival-fm:gpu .

# NVIDIA GPU, development
docker build --build-arg BASE=gpu-base --build-arg INSTALL_DEV=true -t tcga-survival-fm:gpu .
```

**GPU prerequisites** — running the GPU image with `--gpus all` (or the
`pipeline-gpu` compose service) requires:

1. **Native Docker Engine** (`docker-ce`). Docker Desktop for Linux runs the
   engine in a QEMU VM and does **not** support NVIDIA GPU passthrough.
2. **NVIDIA Container Toolkit** on the host:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
3. Verify with: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`.

### Conda

```bash
conda env create -f environment.yml
conda activate tcga-survival-fm
pip install -e "."          # production
pip install -e ".[dev]"    # add dev tools (ruff, mypy, bandit, pre-commit)
```

Enable the Git hook after installing the development tools:

```bash
pre-commit install
```

### pip / venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart: synthetic smoke test

Runs the full train loop end-to-end on synthetic features (no TCGA data).

```bash
# With Docker (CPU)
docker compose run --rm pipeline

# With Docker (GPU) -- requires nvidia-container-toolkit; see Installation > GPU prerequisites
docker compose run --rm pipeline-gpu

# Local
python scripts/make_synthetic_dataset.py --output-dir data/synthetic
tcga-survival train \
  --manifest data/synthetic/manifest.csv \
  --output-dir runs/synthetic \
  --epochs 2 --batch-size 8
```

Outputs land in `runs/synthetic/`: `model.pt`, `metrics.json`, `config.json`.

The synthetic generator accepts `--slides`, `--feature-dim`, `--min-tiles`,
`--max-tiles`, and `--seed`. It creates random feature bags and survival labels;
it is intended only as an end-to-end smoke test.

## Full pipeline with public UNI2-h TCGA embeddings

If you do not have raw WSIs, you can run the full survival pipeline against the
gated [`MahmoodLab/UNI2-h-features`](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features)
dataset (precomputed UNI2-h embeddings for TCGA / CPTAC / PANDA).

### 1. Request HF access
The dataset is gated. Request access on the dataset page using your
**institutional email** (personal `@gmail`/`@hotmail`/`@qq` requests are denied).

### 2. Download a project archive (~10-50 GB per project)
```bash
huggingface-cli login
huggingface-cli download MahmoodLab/UNI2-h-features TCGA/TCGA-BRCA.tar.gz \
  --repo-type dataset --local-dir ~/uni2h
mkdir -p ~/uni2h/TCGA-BRCA && tar -xzf ~/uni2h/TCGA/TCGA-BRCA.tar.gz -C ~/uni2h/TCGA-BRCA
```

### 3. Pull TCGA clinical labels (Docker GPU)

Build the image first if it is not already available:

```bash
docker build --build-arg BASE=gpu-base -t tcga-survival-fm:gpu .
```

```bash
docker run --rm --gpus all -v "$(pwd)/data:/app/data" \
  --user "$(id -u):$(id -g)" tcga-survival-fm:gpu \
  download-clinical --projects TCGA-BRCA \
  --output /app/data/tcga_brca_clinical.csv
```

### 4. Convert .h5 -> .npy + build manifest
```bash
docker run --rm --gpus all --entrypoint="" \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/data:/app/data" \
  -v ~/uni2h/TCGA-BRCA:/h5:ro \
  tcga-survival-fm:gpu \
  python /app/scripts/convert_uni2h_features.py \
    --h5-dir /h5 \
    --clinical-csv /app/data/tcga_brca_clinical.csv \
    --output-dir /app/data/uni2h_brca \
    --max-tiles 2000
```

### 5. Train (Docker GPU)
```bash
docker run --rm --gpus all -v "$(pwd)/data:/app/data" -v "$(pwd)/runs:/app/runs" \
  --user "$(id -u):$(id -g)" tcga-survival-fm:gpu \
  train \
  --manifest /app/data/uni2h_brca/manifest.csv \
  --output-dir /app/runs/uni2h_brca \
  --epochs 50 --batch-size 4 --max-tiles 512 \
  --device cuda
```

Outputs land in `runs/uni2h_brca/`: `model.pt`, `metrics.json`, `config.json`.

## Full workflow (your own WSIs)

### 1. Clinical labels

```bash
tcga-survival download-clinical --output data/tcga_clinical.csv
```

### 2. Tile layout

Place diagnostic tiles one folder per slide. The folder name must start with
the TCGA patient barcode, or the manifest must include an explicit `patient_id`.

```text
data/tiles/
  TCGA-XX-YYYY-01Z-00-DX1/
    tile_000001.png
    tile_000002.png
  TCGA-AA-BBBB-01Z-00-DX1/
    ...
```

### 3. Feature extraction

The default encoder is `MahmoodLab/UNI` with the `timm` backend (requires
`huggingface-cli login` for gated access). Use `owkin/phikon` as an unrestricted
fallback for prototyping. The `transformers` backend is also supported. Use
`--device cuda`, `--device mps`, or `--device cpu` to override automatic device
selection; `auto` prefers MPS, then CUDA, then CPU. Use `--revision` with the
`transformers` backend to pin a model revision.

```bash
tcga-survival featurize-folder \
  --tile-root data/tiles \
  --output-dir data/features \
  --manifest data/manifest.csv \
  --clinical-csv data/tcga_clinical.csv \
  --encoder-name MahmoodLab/UNI \
  --backend timm \
  --batch-size 4 \
  --max-tiles 2000 \
  --device cuda
```

For tight memory: `--batch-size 1 --max-tiles 512`.

Feature extraction writes one float32 `.npy` file per slide directly under
`--output-dir` and writes the manifest to `--manifest`. If a clinical CSV is
provided, its `age_at_diagnosis_days` field is emitted as the optional
`clinical_age_at_diagnosis_days` manifest column.

### 4. Train

```bash
tcga-survival train \
  --manifest data/manifest.csv \
  --output-dir runs/uni_abmil_cox \
  --epochs 50 --batch-size 4 --max-tiles 512 \
  --device cuda
```

Training defaults to 50 epochs, batch size 4, a 20% patient-level validation
split, and at most 512 tiles per slide. Use `--max-tiles all` or `--max-tiles
none` to disable tile subsampling. The run directory contains the best
checkpoint (`model.pt`), its metrics (`metrics.json`), and the resolved
configuration including the selected device (`config.json`).

## Configuration notes

- **Clinical covariates:** manifest columns prefixed with `clinical_` are fed
  to the survival head as numeric features.
- **Splits:** patient-level to prevent slide leakage.
- **Model:** LayerNorm, gated attention MIL pooling, and an MLP risk head. If
  clinical covariates are present, they are encoded and concatenated before the
  risk head.
- **Loss:** negative Cox partial log-likelihood with censoring. Higher predicted
  risk means shorter predicted survival.
- **Metric:** Harrell-style c-index; it may be `NaN` when no comparable observed
  events exist in a validation set.
- **Feature paths:** relative paths are resolved from the manifest directory and
  may not escape that directory.
- **Apple Silicon:** PyTorch uses MPS automatically; the Cox loss falls back to
  CPU for `logcumsumexp`. `PYTORCH_ENABLE_MPS_FALLBACK=1` is pre-set in
  `docker-compose.yml`.

See [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) for the manifest and `.npy`
feature-array contract.

## Command reference

The installed `tcga-survival` command provides:

```text
download-clinical   Fetch GDC survival labels for selected projects.
featurize-folder    Embed image tiles and write feature files plus a manifest.
train               Train and evaluate the attention-MIL Cox model.
recommend-model     Print the recommended encoder and modeling approach.
```

Run `tcga-survival <command> --help` for all options. `download-clinical`
defaults to the supported TCGA solid-tumor projects; pass one or more project
IDs such as `TCGA-BRCA` with `--projects` to restrict the request.

## Development

```bash
ruff check src/ scripts/ tests/    # lint
ruff format src/ scripts/ tests/   # format
bandit -r src/                     # security scan
pytest                             # tests
pre-commit install                 # install git hooks
pre-commit run --all-files         # run hooks manually
```

## Project layout

```
src/tcga_survival/    Package: data, model, losses, metrics, train, cli, gdc
scripts/              Utility scripts (e.g. synthetic dataset generation)
tests/                Unit tests
configs/              Example run configs
docs/                 Extended notes (data schema, model choice)
```

## Disclaimer

Research code. Not a clinical device.
