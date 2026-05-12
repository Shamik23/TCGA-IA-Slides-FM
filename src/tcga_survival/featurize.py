"""Feature extraction from pre-tiled slides with Hugging Face pathology encoders."""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

# Cap PIL image size to mitigate decompression-bomb DoS (200 MP).
PIL_MAX_IMAGE_PIXELS = 200_000_000

# Whitelist for slide_id directory names: alphanumerics, dot, dash, underscore.
_SAFE_SLIDE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# Characters that trigger formula execution in spreadsheet apps when leading.
_CSV_INJECTION_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _validate_slide_id(slide_id: str) -> None:
    if not _SAFE_SLIDE_ID.match(slide_id):
        raise ValueError(
            f"Unsafe slide directory name '{slide_id}'. "
            "Slide IDs must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}."
        )


def _csv_safe(value: str) -> str:
    if value and value[0] in _CSV_INJECTION_PREFIXES:
        return "'" + value
    return value


def list_images(path: str | Path) -> list[Path]:
    folder = Path(path)
    return sorted(
        item
        for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def tcga_patient_id_from_slide_id(slide_id: str) -> str:
    return slide_id[:12] if slide_id.startswith("TCGA-") and len(slide_id) >= 12 else slide_id


def _resolve_device(preferred: str = "auto") -> str:
    import torch

    if preferred != "auto":
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_timm_encoder(model_name: str, device: str):
    import timm
    import torch
    from timm.data import create_transform, resolve_data_config

    hub_name = model_name if model_name.startswith("hf-hub:") else f"hf-hub:{model_name}"
    lower_name = model_name.lower()
    kwargs = {}
    if "mahmoodlab/uni" in lower_name:
        kwargs.update({"init_values": 1e-5, "dynamic_img_size": True})
    if "h0-mini" in lower_name or "virchow" in lower_name:
        kwargs.update({"mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU})

    model = timm.create_model(hub_name, pretrained=True, **kwargs)
    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**config)
    return model, transform


def _load_transformers_encoder(model_name: str, device: str, revision: str | None = None):
    """Load a Hugging Face transformers encoder.

    For supply-chain safety, callers should pin ``revision`` to a known commit
    SHA or tag instead of relying on the default branch.
    """
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision)
    model.eval().to(device)
    return model, processor


def _batched(items: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _embed_with_timm(model, transform, image_paths: Sequence[Path], device: str):
    import torch
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = PIL_MAX_IMAGE_PIXELS
    tensors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            tensors.append(transform(image.convert("RGB")))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        output = model(batch)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 3:
        cfg = getattr(model, "pretrained_cfg", {})
        has_cls = cfg.get("num_classes", 1) != 0 or "cls_token" in str(type(model)).lower()
        if not has_cls:
            raise ValueError(
                f"Model '{type(model).__name__}' produced a 3D output but does not appear to "
                "use a CLS token. Set backend='transformers' or use a ViT-style timm model."
            )
        output = output[:, 0]
    return output.detach().cpu()


def _embed_with_transformers(model, processor, image_paths: Sequence[Path], device: str):
    import torch
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = PIL_MAX_IMAGE_PIXELS
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))

    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        features = output.pooler_output
    else:
        features = output.last_hidden_state[:, 0]
    return features.detach().cpu()


def extract_slide_features(
    tile_dir: str | Path,
    output_path: str | Path,
    encoder_name: str = "MahmoodLab/UNI",
    backend: str = "timm",
    batch_size: int = 4,
    max_tiles: int | None = None,
    device: str = "auto",
    revision: str | None = None,
    tile_seed: int = 0,
) -> tuple[int, int]:
    import numpy as np
    import torch

    resolved_device = _resolve_device(device)
    tile_paths = list_images(tile_dir)
    if max_tiles is not None and len(tile_paths) > max_tiles:
        import random

        rng = random.Random(tile_seed)  # noqa: S311
        tile_paths = sorted(rng.sample(tile_paths, max_tiles))
    if not tile_paths:
        raise ValueError(f"no image tiles found in {tile_dir}")

    if backend == "timm":
        model, processor = _load_timm_encoder(encoder_name, resolved_device)

        def embed(batch: Sequence[Path]):
            return _embed_with_timm(model, processor, batch, resolved_device)
    elif backend == "transformers":
        model, processor = _load_transformers_encoder(
            encoder_name, resolved_device, revision=revision
        )

        def embed(batch: Sequence[Path]):
            return _embed_with_transformers(model, processor, batch, resolved_device)
    else:
        raise ValueError("backend must be 'timm' or 'transformers'")

    feature_batches = []
    for batch in _batched(tile_paths, batch_size):
        feature_batches.append(embed(batch))

    features = torch.cat(feature_batches, dim=0).numpy().astype("float32")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, features)
    return int(features.shape[0]), int(features.shape[1])


def _read_clinical_csv(path: str | Path | None) -> dict[str, dict[str, str]]:
    import warnings

    if path is None:
        return {}

    result: dict[str, dict[str, str]] = {}
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pid = row["patient_id"]
            if pid in result:
                warnings.warn(
                    f"Duplicate patient_id '{pid}' in clinical CSV '{path}'. "
                    "Only the last occurrence will be used.",
                    UserWarning,
                    stacklevel=2,
                )
            result[pid] = row
    return result


def featurize_tile_root(
    tile_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    clinical_csv: str | Path | None = None,
    encoder_name: str = "MahmoodLab/UNI",
    backend: str = "timm",
    batch_size: int = 4,
    max_tiles: int | None = None,
    device: str = "auto",
    revision: str | None = None,
    tile_seed: int = 0,
) -> None:
    root = Path(tile_root)
    output = Path(output_dir)
    clinical = _read_clinical_csv(clinical_csv)
    rows: list[dict[str, str]] = []

    slide_dirs = sorted(item for item in root.iterdir() if item.is_dir() and not item.is_symlink())
    if not slide_dirs:
        raise ValueError(f"no slide directories found in {root}")

    for slide_dir in slide_dirs:
        slide_id = slide_dir.name
        _validate_slide_id(slide_id)
        patient_id = tcga_patient_id_from_slide_id(slide_id)
        feature_path = output / f"{slide_id}.npy"
        slide_hash = int.from_bytes(
            hashlib.blake2b(slide_id.encode("utf-8"), digest_size=4).digest(),
            "big",
        )
        per_slide_seed = (tile_seed + slide_hash) & 0x7FFFFFFF
        tile_count, feature_dim = extract_slide_features(
            tile_dir=slide_dir,
            output_path=feature_path,
            encoder_name=encoder_name,
            backend=backend,
            batch_size=batch_size,
            max_tiles=max_tiles,
            device=device,
            revision=revision,
            tile_seed=per_slide_seed,
        )

        clinical_row = clinical.get(patient_id, {})
        if clinical and not clinical_row:
            import warnings

            warnings.warn(
                f"No clinical data found for patient '{patient_id}' (slide '{slide_id}'). "
                "The manifest will have empty duration_days and event fields for this slide, "
                "which will cause a parsing error during training.",
                UserWarning,
                stacklevel=2,
            )
        row = {
            "patient_id": _csv_safe(patient_id),
            "slide_id": _csv_safe(slide_id),
            "feature_path": _csv_safe(str(feature_path)),
            "duration_days": _csv_safe(clinical_row.get("duration_days", "")),
            "event": _csv_safe(clinical_row.get("event", "")),
            "project_id": _csv_safe(clinical_row.get("project_id", "")),
            "tile_count": str(tile_count),
            "feature_dim": str(feature_dim),
        }
        if clinical_row.get("age_at_diagnosis_days"):
            row["clinical_age_at_diagnosis_days"] = _csv_safe(clinical_row["age_at_diagnosis_days"])
        rows.append(row)
        print(f"{slide_id}: {tile_count} tiles -> {feature_dim}D")

    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["patient_id", "slide_id", "feature_path", "duration_days", "event", "project_id"]
    fieldnames = preferred + [name for name in fieldnames if name not in preferred]

    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
