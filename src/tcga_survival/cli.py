"""Command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "all"}:
        return None
    return _positive_int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcga-survival",
        description="TCGA solid-tumor survival modeling with pathology foundation features.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clinical = subparsers.add_parser("download-clinical", help="Download TCGA survival labels")
    clinical.add_argument("--output", required=True, type=Path)
    clinical.add_argument("--projects", nargs="*", default=None)

    featurize = subparsers.add_parser("featurize-folder", help="Embed tile folders")
    featurize.add_argument("--tile-root", required=True, type=Path)
    featurize.add_argument("--output-dir", required=True, type=Path)
    featurize.add_argument("--manifest", required=True, type=Path)
    featurize.add_argument("--clinical-csv", type=Path)
    featurize.add_argument("--encoder-name", default="MahmoodLab/UNI")
    featurize.add_argument("--backend", choices=["timm", "transformers"], default="timm")
    featurize.add_argument("--batch-size", type=_positive_int, default=4)
    featurize.add_argument("--max-tiles", type=_optional_positive_int, default=None)
    featurize.add_argument("--device", default="auto")

    train = subparsers.add_parser("train", help="Train the attention MIL Cox model")
    train.add_argument("--manifest", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--epochs", type=_positive_int, default=50)
    train.add_argument("--batch-size", type=_positive_int, default=4)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--attention-dim", type=_positive_int, default=256)
    train.add_argument("--hidden-dim", type=_positive_int, default=256)
    train.add_argument("--dropout", type=float, default=0.15)
    train.add_argument("--max-tiles", type=_optional_positive_int, default=512)
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--device", default="auto")

    recommend = subparsers.add_parser("recommend-model", help="Print the model recommendation")
    recommend.set_defaults(recommend_model=True)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download-clinical":
        from .gdc import write_tcga_clinical_csv

        count = write_tcga_clinical_csv(args.output, project_ids=args.projects)
        print(f"wrote {count} patients to {args.output}")
        return

    if args.command == "featurize-folder":
        from .featurize import featurize_tile_root

        featurize_tile_root(
            tile_root=args.tile_root,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            clinical_csv=args.clinical_csv,
            encoder_name=args.encoder_name,
            backend=args.backend,
            batch_size=args.batch_size,
            max_tiles=args.max_tiles,
            device=args.device,
        )
        print(f"wrote manifest to {args.manifest}")
        return

    if args.command == "train":
        from .data import read_manifest
        from .train import train_survival_model

        records = read_manifest(args.manifest)
        metrics = train_survival_model(
            records=records,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            attention_dim=args.attention_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            max_tiles=args.max_tiles,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
        )
        print(f"best metrics: {metrics}")
        return

    if args.command == "recommend-model":
        print(
            "Use MahmoodLab/UNI as a frozen Hugging Face pathology encoder, cache tile "
            "embeddings, and train the included attention MIL Cox survival head. "
            "Use owkin/phikon only as a laptop-friendly prototype fallback for TCGA."
        )
        return

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
