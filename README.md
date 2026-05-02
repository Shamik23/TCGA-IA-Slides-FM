# TCGA Solid-Tumor Survival Foundation Model

This project trains a censored survival model for TCGA solid tumors using frozen
histopathology foundation-model embeddings plus an attention MIL survival head.

## Recommended Hugging Face Model

Use `MahmoodLab/UNI` as the default pathology foundation encoder.

Why this is the right starting point:

- It is a histopathology foundation model that produces slide-tile embeddings
  suitable for downstream whole-slide tasks.
- It is a better scientific fit for TCGA benchmarking than models pretrained on
  TCGA tiles, because the model card states that public datasets such as TCGA,
  CPTAC, PANDA, and TCIA were not used in pretraining.
- A 16 GB MacBook Air should not fine-tune the encoder end to end. Freeze the
  encoder, cache tile embeddings to `.npy`, and train only the attention MIL
  survival head.

If UNI access is unavailable, `owkin/phikon` is a laptop-friendly fallback. Use
it for prototyping, but avoid reporting clean TCGA benchmark claims with it
unless you explicitly account for TCGA being part of its pretraining corpus.

## Project Shape

The core workflow is:

1. Download TCGA clinical survival labels from the GDC API.
2. Prepare diagnostic slide tiles separately, one folder per slide.
3. Use a Hugging Face pathology encoder to cache tile features.
4. Train an attention MIL Cox model on cached features.
5. Evaluate with censored concordance index.

Raw TCGA whole-slide images are very large. This repository intentionally keeps
WSI tiling outside the training loop so the survival model can run on laptop
hardware once features are cached.

## Setup

### Docker (recommended)

```bash
docker compose build
docker compose run --rm tcga-survival --help

# End-to-end synthetic smoke test (writes to ./runs/synthetic):
docker compose run --rm pipeline
```

### Conda

```bash
conda env create -f environment.yml
conda activate tcga-survival-fm
pip install -e ".[dev]"
```

### Local pip / venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For gated Hugging Face models (e.g. `MahmoodLab/UNI`):

```bash
huggingface-cli login            # local
docker compose run --rm tcga-survival huggingface-cli login
```

On Apple Silicon, PyTorch uses MPS automatically. The Cox loss falls back to
CPU for `logcumsumexp`; set `PYTORCH_ENABLE_MPS_FALLBACK=1` to silence the
warning (already set in `docker-compose.yml`).

## Development

```bash
ruff check src/ scripts/ tests/    # lint
ruff format src/ scripts/ tests/   # format
bandit -r src/                     # security scan
pytest                             # unit tests
```

## Data Layout

Clinical labels:

```bash
tcga-survival download-clinical --output data/tcga_clinical.csv
```

Expected tile layout:

```text
data/tiles/
  TCGA-XX-YYYY-01Z-00-DX1/
    tile_000001.png
    tile_000002.png
  TCGA-AA-BBBB-01Z-00-DX1/
    tile_000001.png
```

Each slide folder name should start with the TCGA patient barcode, or be listed
in a manifest with an explicit `patient_id`.

## Feature Extraction

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

For low-memory runs, use `--batch-size 1` or `--max-tiles 512`.

## Train

```bash
tcga-survival train \
  --manifest data/manifest.csv \
  --output-dir runs/uni_abmil_cox \
  --epochs 50 \
  --batch-size 4 \
  --max-tiles 512
```

Outputs include:

- `model.pt`: best checkpoint by validation c-index
- `metrics.json`: validation loss and c-index
- `config.json`: run configuration

## Synthetic Smoke Test

Use this when dependencies are installed but TCGA data is not ready yet:

```bash
python scripts/make_synthetic_dataset.py --output-dir data/synthetic
tcga-survival train \
  --manifest data/synthetic/manifest.csv \
  --output-dir runs/synthetic \
  --epochs 2 \
  --batch-size 8
```

## Notes

- The survival head uses Cox partial likelihood and handles censoring.
- The split is patient-level to avoid slide leakage.
- The code accepts optional numeric clinical covariates using manifest columns
  prefixed with `clinical_`.
- This is research code, not a clinical device.
