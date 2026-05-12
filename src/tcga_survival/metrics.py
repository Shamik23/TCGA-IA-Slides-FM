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
    raw_events = [float(value) for value in event]
    events: list[int] = []
    for value in raw_events:
        if value not in (0.0, 1.0):
            raise ValueError(
                f"event values must be binary (0 or 1); got {value!r}. "
                "Pass observed event indicators, not probabilities."
            )
        events.append(int(value))

    if not (len(durations) == len(risks) == len(events)):
        raise ValueError(
            f"duration, risk, and event must have the same length; "
            f"got {len(durations)}, {len(risks)}, {len(events)}"
        )

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
