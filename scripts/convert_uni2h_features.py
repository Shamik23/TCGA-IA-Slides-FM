#!/usr/bin/env python3
"""Convert MahmoodLab/UNI2-h-features `.h5` files into the manifest + per-slide
`.npy` layout consumed by `tcga-survival train`.

Input layout (after `tar -xzf TCGA-BRCA.tar.gz`):
    <h5-dir>/
        TCGA-XX-YYYY-01Z-00-DX1.h5
        TCGA-AA-BBBB-01Z-00-DX1.h5
        ...

Each `.h5` contains:
    features: shape (1, N, 1536) -- UNI2-h embeddings, N = number of patches
    coords:   shape (1, N, 2)    -- patch (x, y) coordinates (ignored here)

Output layout:
    <output-dir>/features/<slide_id>.npy   # shape (N, 1536)
    <output-dir>/manifest.csv              # joined with clinical labels

Usage:
    python scripts/convert_uni2h_features.py \\
        --h5-dir /path/to/extracted/TCGA-BRCA \\
        --clinical-csv data/tcga_brca_clinical.csv \\
        --output-dir data/uni2h_brca \\
        [--limit 16] [--max-tiles 2000]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

# Allow running this script without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcga_survival.featurize import tcga_patient_id_from_slide_id  # noqa: E402


def _slide_id_from_filename(path: Path) -> str:
    """Strip the trailing .h5 (and any .svs.h5 / .tiff.h5 wrapper)."""
    name = path.name
    if name.endswith(".h5"):
        name = name[:-3]
    for suffix in (".svs", ".tiff", ".tif"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _load_features(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        if "features" not in handle:
            raise ValueError(f"{h5_path} has no 'features' dataset")
        features = handle["features"][:]
    features = np.asarray(features)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if features.ndim != 2:
        raise ValueError(
            f"{h5_path}: expected features of shape (N, D) or (1, N, D); got {features.shape}"
        )
    return features.astype(np.float32, copy=False)


def _read_clinical(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "patient_id" not in reader.fieldnames:
            raise ValueError(f"{csv_path} missing 'patient_id' column")
        return {row["patient_id"]: row for row in reader}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--h5-dir", type=Path, required=True, help="Directory of UNI2-h .h5 files")
    parser.add_argument(
        "--clinical-csv",
        type=Path,
        required=True,
        help="CSV from `tcga-survival download-clinical`",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Where to write features/ and manifest.csv"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process at most N slides (smoke testing)"
    )
    parser.add_argument(
        "--max-tiles", type=int, default=None, help="Subsample to N tiles per slide (deterministic)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for tile subsampling")
    args = parser.parse_args()

    h5_dir: Path = args.h5_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    if not h5_dir.is_dir():
        parser.error(f"--h5-dir does not exist or is not a directory: {h5_dir}")
    clinical_by_patient = _read_clinical(args.clinical_csv)

    h5_files = sorted(h5_dir.glob("*.h5"))
    if args.limit is not None:
        h5_files = h5_files[: args.limit]
    if not h5_files:
        parser.error(f"no .h5 files found in {h5_dir}")

    rows: list[dict[str, str]] = []
    skipped_no_clinical = 0
    skipped_invalid_id = 0
    rng = np.random.default_rng(args.seed)

    for h5_path in h5_files:
        slide_id = _slide_id_from_filename(h5_path)
        try:
            patient_id = tcga_patient_id_from_slide_id(slide_id)
        except ValueError:
            skipped_invalid_id += 1
            continue

        clinical = clinical_by_patient.get(patient_id)
        if clinical is None:
            skipped_no_clinical += 1
            continue

        features = _load_features(h5_path)
        if args.max_tiles is not None and features.shape[0] > args.max_tiles:
            indices = rng.choice(features.shape[0], size=args.max_tiles, replace=False)
            indices.sort()
            features = features[indices]

        feature_path = feature_dir / f"{slide_id}.npy"
        np.save(feature_path, features)

        rows.append(
            {
                "patient_id": patient_id,
                "slide_id": slide_id,
                "feature_path": str(feature_path.relative_to(output_dir)),
                "duration_days": clinical["duration_days"],
                "event": clinical["event"],
                "project_id": clinical.get("project_id", ""),
            }
        )

    if not rows:
        print("ERROR: no slides matched the clinical CSV; manifest not written.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["patient_id", "slide_id", "feature_path", "duration_days", "event", "project_id"]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} slides to {manifest_path}")
    if skipped_no_clinical:
        print(f"  skipped {skipped_no_clinical} slides with no matching clinical record")
    if skipped_invalid_id:
        print(f"  skipped {skipped_invalid_id} slides with unparseable TCGA barcode")


if __name__ == "__main__":
    main()
