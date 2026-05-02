"""Evaluation metrics for censored survival prediction."""

from __future__ import annotations

from collections.abc import Sequence


def concordance_index(
    duration: Sequence[float], risk: Sequence[float], event: Sequence[int]
) -> float:
    """Harrell-style c-index where larger risk means shorter survival."""

    import math

    durations = [float(value) for value in duration]
    risks = [float(value) for value in risk]
    events = [int(value) for value in event]

    concordant = 0.0
    comparable = 0
    n = len(durations)

    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if durations[i] >= durations[j]:
                continue
            comparable += 1
            if risks[i] > risks[j]:
                concordant += 1.0
            elif risks[i] == risks[j]:
                concordant += 0.5

    if comparable == 0:
        return math.nan
    return concordant / comparable
