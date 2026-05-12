#!/usr/bin/env python3
"""Create a small synthetic cached-feature dataset for smoke testing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slides", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--min-tiles", type=int, default=32)
    parser.add_argument("--max-tiles", type=int, default=96)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.slides <= 0:
        parser.error("--slides must be a positive integer")
    if args.min_tiles > args.max_tiles:
        parser.error("--min-tiles must be <= --max-tiles")

    rng = np.random.default_rng(args.seed)
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    signal = rng.normal(size=args.feature_dim).astype("float32")
    signal /= np.linalg.norm(signal)

    for idx in range(args.slides):
        patient_id = f"SYN-{idx:04d}"
        slide_id = f"{patient_id}-SLIDE"
        tile_count = int(rng.integers(args.min_tiles, args.max_tiles + 1))
        features = rng.normal(size=(tile_count, args.feature_dim)).astype("float32")
        latent_risk = float(features.mean(axis=0).dot(signal))
        duration = max(10, int(365 * np.exp(-latent_risk + rng.normal(scale=0.3))))
        event = int(rng.random() > 0.25)

        feature_path = feature_dir / f"{slide_id}.npy"
        np.save(feature_path, features)
        rows.append(
            {
                "patient_id": patient_id,
                "slide_id": slide_id,
                "feature_path": str(feature_path.relative_to(args.output_dir)),
                "duration_days": str(duration),
                "event": str(event),
                "project_id": "SYNTHETIC",
            }
        )

    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} slides to {manifest}")


if __name__ == "__main__":
    main()
