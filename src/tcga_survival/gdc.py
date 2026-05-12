"""Small GDC API client for TCGA clinical survival labels."""

from __future__ import annotations

import csv
import json
import ssl
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path

HTTP_TIMEOUT_SECONDS = 30

GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"

TCGA_SOLID_TUMOR_PROJECTS = [
    "TCGA-ACC",
    "TCGA-BLCA",
    "TCGA-BRCA",
    "TCGA-CESC",
    "TCGA-CHOL",
    "TCGA-COAD",
    "TCGA-ESCA",
    "TCGA-GBM",
    "TCGA-HNSC",
    "TCGA-KICH",
    "TCGA-KIRC",
    "TCGA-KIRP",
    "TCGA-LGG",
    "TCGA-LIHC",
    "TCGA-LUAD",
    "TCGA-LUSC",
    "TCGA-MESO",
    "TCGA-OV",
    "TCGA-PAAD",
    "TCGA-PCPG",
    "TCGA-PRAD",
    "TCGA-READ",
    "TCGA-SARC",
    "TCGA-SKCM",
    "TCGA-STAD",
    "TCGA-TGCT",
    "TCGA-THCA",
    "TCGA-THYM",
    "TCGA-UCEC",
    "TCGA-UCS",
    "TCGA-UVM",
]


FIELDS = [
    "submitter_id",
    "project.project_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "diagnoses.days_to_death",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.age_at_diagnosis",
    "diagnoses.days_to_last_known_disease_status",
]


def _first_numeric(values: Iterable[object]) -> float | None:
    for value in values:
        if value in (None, "", "not reported", "Not Reported"):
            continue
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def _extract_case(row: dict[str, object]) -> dict[str, str] | None:
    demographic_raw = row.get("demographic") or {}
    project_raw = row.get("project") or {}
    diagnoses_raw = row.get("diagnoses") or []
    demographic: dict[str, object] = demographic_raw if isinstance(demographic_raw, dict) else {}
    project: dict[str, object] = project_raw if isinstance(project_raw, dict) else {}
    diagnoses: list[dict[str, object]] = (
        [d for d in diagnoses_raw if isinstance(d, dict)] if isinstance(diagnoses_raw, list) else []
    )

    vital_status = str(demographic.get("vital_status", "")).lower()
    diagnosis_days_to_death = [diagnosis.get("days_to_death") for diagnosis in diagnoses]
    days_to_death = _first_numeric([demographic.get("days_to_death")] + diagnosis_days_to_death)
    days_to_last_follow_up = _first_numeric(
        [diagnosis.get("days_to_last_follow_up") for diagnosis in diagnoses]
        + [diagnosis.get("days_to_last_known_disease_status") for diagnosis in diagnoses]
    )
    age_at_diagnosis = _first_numeric(
        [diagnosis.get("age_at_diagnosis") for diagnosis in diagnoses]
    )

    # vital_status is authoritative; only fall back to days_to_death heuristic
    # when vital_status is missing or unknown.
    if vital_status == "dead":
        event = 1
    elif vital_status == "alive":
        event = 0
    else:
        event = 1 if days_to_death is not None else 0
    duration = days_to_death if event else days_to_last_follow_up
    if duration is None or duration <= 0:
        return None

    return {
        "patient_id": str(row.get("submitter_id", "")),
        "project_id": str(project.get("project_id", "")),
        "duration_days": str(int(duration)),
        "event": str(event),
        "age_at_diagnosis_days": "" if age_at_diagnosis is None else str(int(age_at_diagnosis)),
    }


def fetch_tcga_clinical_survival(
    project_ids: list[str] | None = None,
    page_size: int = 2000,
) -> list[dict[str, str]]:
    projects = project_ids or TCGA_SOLID_TUMOR_PROJECTS
    filters = {
        "op": "in",
        "content": {"field": "project.project_id", "value": projects},
    }
    rows = []
    offset = 0
    total = None

    while total is None or offset < total:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(FIELDS),
            "format": "JSON",
            "size": str(page_size),
            "from": str(offset),
        }
        url = f"{GDC_CASES_ENDPOINT}?{urllib.parse.urlencode(params)}"
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"refusing to fetch non-http(s) URL: {url}")

        ssl_ctx = ssl.create_default_context()
        with urllib.request.urlopen(  # noqa: S310  # nosec B310 - scheme validated http(s) above
            url, timeout=HTTP_TIMEOUT_SECONDS, context=ssl_ctx
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))

        try:
            data = payload["data"]
            total = int(data["pagination"]["total"])
            cases = data["hits"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Unexpected GDC API response structure at offset {offset}: {exc}. "
                f"Response keys: {list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
            ) from exc
        if not cases:
            break

        for case in cases:
            extracted = _extract_case(case)
            if extracted is not None:
                rows.append(extracted)
        offset += len(cases)

    return rows


def write_tcga_clinical_csv(output_path: str | Path, project_ids: list[str] | None = None) -> int:
    rows = fetch_tcga_clinical_survival(project_ids=project_ids)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "patient_id",
        "project_id",
        "duration_days",
        "event",
        "age_at_diagnosis_days",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
