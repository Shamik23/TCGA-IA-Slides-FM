"""Survival losses."""

from __future__ import annotations


def cox_partial_log_likelihood(risk, duration, event):
    """Negative Cox partial log-likelihood.

    Higher risk means earlier predicted event. Durations are sorted descending so
    the cumulative denominator contains the risk set for each observed event.
    """

    import torch

    risk = risk.reshape(-1)
    duration = duration.reshape(-1)
    event = event.reshape(-1).float()

    order = torch.argsort(duration, descending=True)
    risk = risk[order]
    event = event[order]

    log_risk_set = torch.logcumsumexp(risk, dim=0)
    observed = event.sum()
    if observed.item() == 0:
        return risk.sum() * 0.0

    return -((risk - log_risk_set) * event).sum() / observed.clamp_min(1.0)


cox_ph_loss = cox_partial_log_likelihood

