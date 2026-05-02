from pathlib import Path

from tcga_survival.data import SlideRecord, patient_level_split


def test_patient_level_split_keeps_patient_slides_together():
    records = [
        SlideRecord("P1", "S1", Path("a.npy"), 10, 1),
        SlideRecord("P1", "S2", Path("b.npy"), 11, 0),
        SlideRecord("P2", "S3", Path("c.npy"), 12, 1),
        SlideRecord("P3", "S4", Path("d.npy"), 13, 0),
    ]

    train, val = patient_level_split(records, val_fraction=0.34, seed=1)
    train_patients = {record.patient_id for record in train}
    val_patients = {record.patient_id for record in val}

    assert train_patients.isdisjoint(val_patients)
    assert train
    assert val
