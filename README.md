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

```bash
docker compose build
docker compose run --rm tcga-survival --help
```

### Conda

```bash
conda env create -f environment.yml
conda activate tcga-survival-fm
pip install -e ".[dev]"
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
# With Docker
docker compose run --rm pipeline

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
