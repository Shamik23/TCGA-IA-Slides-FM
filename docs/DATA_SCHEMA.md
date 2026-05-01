# Data Schema

## Manifest

Training expects a CSV manifest with one row per slide:

```csv
patient_id,slide_id,feature_path,duration_days,event,project_id
TCGA-XX-YYYY,TCGA-XX-YYYY-01Z-00-DX1,data/features/TCGA-XX-YYYY-01Z-00-DX1.npy,481,1,TCGA-BRCA
```

Required columns:

- `patient_id`: TCGA case barcode, normally the first 12 characters.
- `slide_id`: unique slide identifier.
- `feature_path`: path to a `.npy` array with shape `[tiles, feature_dim]`.
- `duration_days`: observed survival or follow-up time in days.
- `event`: `1` for death, `0` for censored.

Optional columns:

- `project_id`: TCGA cancer project, such as `TCGA-BRCA`.
- Any numeric column prefixed with `clinical_`, for example
  `clinical_age_at_diagnosis_days`.

## Feature Arrays

Each feature file should contain either:

- `[num_tiles, feature_dim]` for a slide bag, or
- `[feature_dim]` for a pre-pooled slide vector.

Arrays are loaded as `float32`.

