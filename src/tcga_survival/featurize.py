"""Feature extraction from pre-tiled slides with Hugging Face pathology encoders."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def list_images(path: str | Path) -> List[Path]:
    folder = Path(path)
    return sorted(
        item for item in folder.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
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


def _load_transformers_encoder(model_name: str, device: str):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    return model, processor


def _batched(items: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _embed_with_timm(model, transform, image_paths: Sequence[Path], device: str):
    import torch
    from PIL import Image

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
        output = output[:, 0]
    return output.detach().cpu()


def _embed_with_transformers(model, processor, image_paths: Sequence[Path], device: str):
    import torch
    from PIL import Image

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
    max_tiles: Optional[int] = None,
    device: str = "auto",
) -> Tuple[int, int]:
    import numpy as np
    import torch

    resolved_device = _resolve_device(device)
    tile_paths = list_images(tile_dir)
    if max_tiles is not None:
        tile_paths = tile_paths[:max_tiles]
    if not tile_paths:
        raise ValueError(f"no image tiles found in {tile_dir}")

    if backend == "timm":
        model, processor = _load_timm_encoder(encoder_name, resolved_device)
        embed = lambda batch: _embed_with_timm(model, processor, batch, resolved_device)
    elif backend == "transformers":
        model, processor = _load_transformers_encoder(encoder_name, resolved_device)
        embed = lambda batch: _embed_with_transformers(model, processor, batch, resolved_device)
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


def _read_clinical_csv(path: Optional[str | Path]) -> Dict[str, Dict[str, str]]:
    if path is None:
        return {}

    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["patient_id"]: row for row in reader}


def featurize_tile_root(
    tile_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    clinical_csv: Optional[str | Path] = None,
    encoder_name: str = "MahmoodLab/UNI",
    backend: str = "timm",
    batch_size: int = 4,
    max_tiles: Optional[int] = None,
    device: str = "auto",
) -> None:
    root = Path(tile_root)
    output = Path(output_dir)
    clinical = _read_clinical_csv(clinical_csv)
    rows: List[Dict[str, str]] = []

    slide_dirs = sorted(item for item in root.iterdir() if item.is_dir())
    if not slide_dirs:
        raise ValueError(f"no slide directories found in {root}")

    for slide_dir in slide_dirs:
        slide_id = slide_dir.name
        patient_id = tcga_patient_id_from_slide_id(slide_id)
        feature_path = output / f"{slide_id}.npy"
        tile_count, feature_dim = extract_slide_features(
            tile_dir=slide_dir,
            output_path=feature_path,
            encoder_name=encoder_name,
            backend=backend,
            batch_size=batch_size,
            max_tiles=max_tiles,
            device=device,
        )

        clinical_row = clinical.get(patient_id, {})
        row = {
            "patient_id": patient_id,
            "slide_id": slide_id,
            "feature_path": str(feature_path),
            "duration_days": clinical_row.get("duration_days", ""),
            "event": clinical_row.get("event", ""),
            "project_id": clinical_row.get("project_id", ""),
            "tile_count": str(tile_count),
            "feature_dim": str(feature_dim),
        }
        if clinical_row.get("age_at_diagnosis_days"):
            row["clinical_age_at_diagnosis_days"] = clinical_row["age_at_diagnosis_days"]
        rows.append(row)
        print(f"{slide_id}: {tile_count} tiles -> {feature_dim}D")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = ["patient_id", "slide_id", "feature_path", "duration_days", "event", "project_id"]
    fieldnames = preferred + [name for name in fieldnames if name not in preferred]

    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

