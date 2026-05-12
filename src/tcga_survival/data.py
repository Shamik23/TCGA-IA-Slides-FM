"""Dataset utilities for slide-level survival modeling."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

CLINICAL_PREFIX = "clinical_"


@dataclass(frozen=True)
class SlideRecord:
    patient_id: str
    slide_id: str
    feature_path: Path
    duration_days: float
    event: int
    project_id: str = ""
    clinical: tuple[float, ...] = ()


def _as_float(value: object, default: float | None = None) -> float:
    if value is None or value == "":
        if default is None:
            raise ValueError("missing required numeric value")
        return default
    return float(value)  # type: ignore[arg-type]


def _as_event(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "dead", "deceased"}:
        return 1
    if text in {"0", "false", "no", "alive", "censored"}:
        return 0
    raise ValueError(f"could not parse event value: {value!r}")


def read_manifest(path: str | Path) -> list[SlideRecord]:
    manifest_path = Path(path)
    records: list[SlideRecord] = []

    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{manifest_path} has no header row")

        clinical_fields = [name for name in reader.fieldnames if name.startswith(CLINICAL_PREFIX)]
        for row in reader:
            duration_key = "duration_days" if "duration_days" in row else "duration"
            feature_path = Path(row["feature_path"])
            if not feature_path.is_absolute():
                feature_path = manifest_path.parent / feature_path
            feature_path = feature_path.resolve()
            manifest_root = manifest_path.parent.resolve()
            try:
                feature_path.relative_to(manifest_root)
            except ValueError as exc:
                raise ValueError(
                    f"feature_path '{feature_path}' escapes the manifest directory '{manifest_root}'. "
                    "Use absolute paths or paths relative to the manifest file."
                ) from exc

            duration = _as_float(row.get(duration_key))
            if duration <= 0:
                raise ValueError(f"duration_days must be positive for slide {row.get('slide_id')}")

            clinical = tuple(_as_float(row.get(name), default=0.0) for name in clinical_fields)
            records.append(
                SlideRecord(
                    patient_id=row["patient_id"],
                    slide_id=row["slide_id"],
                    feature_path=feature_path,
                    duration_days=duration,
                    event=_as_event(row["event"]),
                    project_id=row.get("project_id", ""),
                    clinical=clinical,
                )
            )

    if not records:
        raise ValueError(f"{manifest_path} did not contain any slide records")
    return records


def infer_feature_dim(records: Sequence[SlideRecord]) -> int:
    import numpy as np

    if not records:
        raise ValueError("cannot infer feature dimension from an empty record list")
    array = np.load(records[0].feature_path, mmap_mode="r", allow_pickle=False)
    try:
        if array.ndim == 1:
            return int(array.shape[0])
        if array.ndim == 2:
            return int(array.shape[1])
        raise ValueError(f"expected 1D or 2D feature array, got shape {array.shape}")
    finally:
        # Close memory-mapped backing file deterministically.
        if hasattr(array, "_mmap") and array._mmap is not None:
            array._mmap.close()
        del array


def patient_level_split(
    records: Sequence[SlideRecord], val_fraction: float = 0.2, seed: int = 13
) -> tuple[list[SlideRecord], list[SlideRecord]]:
    import random

    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")

    patient_ids = sorted({record.patient_id for record in records})
    # nosec B311 - deterministic shuffle for reproducible splits, not security-sensitive
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(patient_ids)

    if len(patient_ids) <= 1 or val_fraction == 0:
        return list(records), []

    val_count = max(1, int(round(len(patient_ids) * val_fraction)))
    val_patients = set(patient_ids[:val_count])

    train_records = [record for record in records if record.patient_id not in val_patients]
    val_records = [record for record in records if record.patient_id in val_patients]
    if not train_records:
        raise ValueError("validation split consumed all records; lower val_fraction")
    return train_records, val_records


class SlideBagDataset:
    """PyTorch dataset for cached variable-length slide feature bags."""

    def __init__(
        self,
        records: Sequence[SlideRecord],
        max_tiles: int | None = None,
        seed: int = 13,
    ) -> None:
        self.records = list(records)
        self.max_tiles = max_tiles
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        import numpy as np

        record = self.records[index]
        features = np.load(record.feature_path, allow_pickle=False).astype("float32", copy=False)
        if features.ndim == 1:
            features = features[None, :]
        if features.ndim != 2:
            raise ValueError(f"{record.feature_path} must be 1D or 2D, got {features.shape}")

        if self.max_tiles is not None and features.shape[0] > self.max_tiles:
            rng = np.random.default_rng(self.seed + index)
            indices = rng.choice(features.shape[0], size=self.max_tiles, replace=False)
            indices.sort()
            features = features[indices]

        return {
            "features": features,
            "duration": float(record.duration_days),
            "event": int(record.event),
            "clinical": record.clinical,
            "patient_id": record.patient_id,
            "slide_id": record.slide_id,
        }


def collate_slide_bags(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    import torch

    max_tiles = max(item["features"].shape[0] for item in batch)  # type: ignore[attr-defined]
    feature_dim = batch[0]["features"].shape[1]  # type: ignore[attr-defined]
    clinical_dim = len(batch[0]["clinical"])  # type: ignore[arg-type]
    for i, item in enumerate(batch):
        item_clinical_dim = len(item["clinical"])  # type: ignore[arg-type]
        if item_clinical_dim != clinical_dim:
            raise ValueError(
                f"Inconsistent clinical dimension in batch: item 0 has {clinical_dim} features, "
                f"item {i} has {item_clinical_dim}. All records must have the same clinical columns."
            )

    features = torch.zeros(len(batch), max_tiles, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_tiles, dtype=torch.bool)
    durations = torch.zeros(len(batch), dtype=torch.float32)
    events = torch.zeros(len(batch), dtype=torch.float32)
    clinical = torch.zeros(len(batch), clinical_dim, dtype=torch.float32)

    patient_ids: list[str] = []
    slide_ids: list[str] = []
    for row, item in enumerate(batch):
        item_features = torch.as_tensor(item["features"], dtype=torch.float32)
        tile_count = item_features.shape[0]
        features[row, :tile_count] = item_features
        mask[row, :tile_count] = True
        durations[row] = float(item["duration"])  # type: ignore[arg-type]
        events[row] = float(item["event"])  # type: ignore[arg-type]
        if clinical_dim:
            clinical[row] = torch.as_tensor(item["clinical"], dtype=torch.float32)
        patient_ids.append(str(item["patient_id"]))
        slide_ids.append(str(item["slide_id"]))

    return {
        "features": features,
        "mask": mask,
        "duration": durations,
        "event": events,
        "clinical": clinical,
        "patient_id": patient_ids,
        "slide_id": slide_ids,
    }


def group_by_patient(records: Iterable[SlideRecord]) -> dict[str, list[SlideRecord]]:
    grouped: dict[str, list[SlideRecord]] = {}
    for record in records:
        grouped.setdefault(record.patient_id, []).append(record)
    return grouped
