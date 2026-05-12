# TCGA Survival Foundation Model

Train a censored survival model for TCGA solid tumors on frozen histopathology
foundation-model embeddings with an attention-MIL Cox head.

## Pipeline

1. Fetch TCGA clinical survival labels from the GDC API.
2. Tile diagnostic slides into per-slide folders (external to this repo).
3. Cache tile features with a frozen pathology encoder.
4. Train the attention-MIL Cox head on cached features.
5. Evaluate with censored concordance index (c-index).

Tiling is kept outside the training loop so the model trains on laptop hardware
once features are cached.

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
docker build --build-arg BASE=gpu-base -t tcga-survival-fm:latest .

# NVIDIA GPU, development
docker build --build-arg BASE=gpu-base --build-arg INSTALL_DEV=true -t tcga-survival-fm:latest .
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

## Full workflow

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

Default encoder is `MahmoodLab/UNI` (requires `huggingface-cli login` for gated
access). Use `owkin/phikon` as an unrestricted fallback for prototyping.

```bash
tcga-survival featurize-folder \
  --tile-root data/tiles \
  --output-dir data/features \
  --manifest data/manifest.csv \
  --clinical-csv data/tcga_clinical.csv \
  --encoder-name MahmoodLab/UNI \
  --backend timm \
  --batch-size 4 \
  --max-tiles 2000
```

For tight memory: `--batch-size 1 --max-tiles 512`.

### 4. Train

```bash
tcga-survival train \
  --manifest data/manifest.csv \
  --output-dir runs/uni_abmil_cox \
  --epochs 50 --batch-size 4 --max-tiles 512
```

## Configuration notes

- **Clinical covariates:** manifest columns prefixed with `clinical_` are fed
  to the survival head as numeric features.
- **Splits:** patient-level to prevent slide leakage.
- **Loss:** Cox partial log-likelihood with censoring.
- **Apple Silicon:** PyTorch uses MPS automatically; the Cox loss falls back to
  CPU for `logcumsumexp`. `PYTORCH_ENABLE_MPS_FALLBACK=1` is pre-set in
  `docker-compose.yml`.

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
