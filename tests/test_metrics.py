from tcga_survival.metrics import concordance_index


def test_concordance_index_perfect_ordering():
    duration = [5, 10, 20]
    event = [1, 1, 0]
    risk = [3.0, 2.0, 1.0]

    assert concordance_index(duration, risk, event) == 1.0


def test_concordance_index_handles_no_comparable_pairs():
    import math

    result = concordance_index([5, 6], [0.2, 0.1], [0, 0])
    assert math.isnan(result)
