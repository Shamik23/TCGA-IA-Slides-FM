"""Training loop for cached TCGA slide features."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .data import (
    SlideBagDataset,
    SlideRecord,
    collate_slide_bags,
    infer_feature_dim,
    patient_level_split,
)
from .losses import cox_ph_loss
from .metrics import concordance_index
from .model import CoxMILSurvivalModel


def resolve_device(preferred: str = "auto") -> str:
    import torch

    if preferred != "auto":
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _move_batch(batch: dict[str, object], device: str) -> dict[str, object]:
    moved = dict(batch)
    for key in ("features", "mask", "duration", "event", "clinical"):
        moved[key] = batch[key].to(device)  # type: ignore[attr-defined]
    return moved


def _run_epoch(model, loader, optimizer, device: str) -> float:
    import torch

    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["features"], batch["mask"], batch["clinical"])
        loss = cox_ph_loss(output["risk"], batch["duration"], batch["event"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        count = int(batch["features"].shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_count += count

    return total_loss / max(total_count, 1)


def evaluate(model, loader, device: str) -> dict[str, float]:
    import torch

    model.eval()
    losses: list[float] = []
    risks: list[float] = []
    durations: list[float] = []
    events: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(batch["features"], batch["mask"], batch["clinical"])
            loss = cox_ph_loss(output["risk"], batch["duration"], batch["event"])
            losses.append(float(loss.cpu()))
            risks.extend(float(value) for value in output["risk"].detach().cpu())
            durations.extend(float(value) for value in batch["duration"].detach().cpu())
            events.extend(int(value) for value in batch["event"].detach().cpu())

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "c_index": concordance_index(durations, risks, events),
    }


def build_loaders(
    train_records: Sequence[SlideRecord],
    val_records: Sequence[SlideRecord],
    batch_size: int,
    max_tiles: int | None,
    seed: int,
) -> tuple[object, object | None]:
    from torch.utils.data import DataLoader

    train_dataset = SlideBagDataset(train_records, max_tiles=max_tiles, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_slide_bags,
        num_workers=0,
    )

    val_loader = None
    if val_records:
        val_dataset = SlideBagDataset(val_records, max_tiles=max_tiles, seed=seed)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_slide_bags,
            num_workers=0,
        )

    return train_loader, val_loader


def train_survival_model(
    records: Sequence[SlideRecord],
    output_dir: str | Path,
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    attention_dim: int = 256,
    hidden_dim: int = 256,
    dropout: float = 0.15,
    max_tiles: int | None = 512,
    val_fraction: float = 0.2,
    seed: int = 13,
    device: str = "auto",
) -> dict[str, float]:
    import torch

    torch.manual_seed(seed)
    resolved_device = resolve_device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_records, val_records = patient_level_split(records, val_fraction=val_fraction, seed=seed)
    feature_dim = infer_feature_dim(records)
    clinical_dim = len(records[0].clinical)

    train_loader, val_loader = build_loaders(
        train_records,
        val_records,
        batch_size=batch_size,
        max_tiles=max_tiles,
        seed=seed,
    )

    import torch.nn as nn

    model = cast(
        nn.Module,
        CoxMILSurvivalModel(
            feature_dim=feature_dim,
            clinical_dim=clinical_dim,
            attention_dim=attention_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ),
    ).to(resolved_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_metric = float("-inf")
    best_metrics: dict[str, float] = {"c_index": float("nan"), "loss": float("nan")}

    run_config = {
        "feature_dim": feature_dim,
        "clinical_dim": clinical_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "attention_dim": attention_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "max_tiles": max_tiles,
        "val_fraction": val_fraction,
        "seed": seed,
        "device": resolved_device,
        "train_slides": len(train_records),
        "val_slides": len(val_records),
    }
    (output_path / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, resolved_device)
        if val_loader is not None:
            metrics = evaluate(model, val_loader, resolved_device)
        else:
            metrics = {"loss": train_loss, "c_index": float("nan")}

        metrics["train_loss"] = train_loss
        metrics["epoch"] = float(epoch)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} c_index={metrics['c_index']:.4f}"
        )

        score = metrics["c_index"]
        if score != score:
            score = -metrics["loss"]
        if score > best_metric:
            best_metric = score
            best_metrics = metrics
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": run_config,
                    "metrics": best_metrics,
                },
                output_path / "model.pt",
            )

    (output_path / "metrics.json").write_text(json.dumps(best_metrics, indent=2) + "\n")
    return best_metrics
