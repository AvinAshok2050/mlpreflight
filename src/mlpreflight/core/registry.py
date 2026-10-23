"""Check registration and deterministic readiness scoring."""
from __future__ import annotations

from typing import Callable

from .context import InspectionContext
from .finding import Finding, Severity

CheckFn = Callable[[InspectionContext], list[Finding]]

_REGISTRY: list[tuple[str, CheckFn]] = []

# Score model -------------------------------------------------------------
# Deterministic and reconstructable by hand: each finding deducts
# weight * PENALTY[severity] points from 100. Any CRITICAL additionally caps
# the final score, so a dataset with a hard blocker can never look healthy.
PENALTY = {Severity.CRITICAL: 18.0, Severity.WARNING: 6.0, Severity.PASSED: 0.0,
           Severity.SKIPPED: 0.0}
CRITICAL_CAP = 60.0
WARNING_ONLY_CAP = 90.0


def register(check_id: str) -> Callable[[CheckFn], CheckFn]:
    def deco(fn: CheckFn) -> CheckFn:
        _REGISTRY.append((check_id, fn))
        return fn
    return deco


def all_checks() -> list[tuple[str, CheckFn]]:
    return list(_REGISTRY)


def compute_score(findings: list[Finding]) -> tuple[int, list[dict]]:
    """Return (score, breakdown). Breakdown explains every point deducted."""
    score = 100.0
    breakdown: list[dict] = []

    for f in findings:
        penalty = PENALTY[f.severity] * f.weight * f.confidence
        if penalty > 0:
            score -= penalty
            breakdown.append({
                "check_id": f.check_id,
                "severity": f.severity.value,
                "weight": f.weight,
                "confidence": round(f.confidence, 2),
                "points_lost": round(penalty, 1),
            })

    has_critical = any(f.severity is Severity.CRITICAL for f in findings)
    has_warning = any(f.severity is Severity.WARNING for f in findings)

    if has_critical and score > CRITICAL_CAP:
        breakdown.append({
            "check_id": "__cap__",
            "severity": "critical",
            "note": f"score capped at {CRITICAL_CAP:.0f} because a Critical finding exists",
            "points_lost": round(score - CRITICAL_CAP, 1),
        })
        score = CRITICAL_CAP
    elif has_warning and score > WARNING_ONLY_CAP:
        score = WARNING_ONLY_CAP

    return max(0, int(round(score))), breakdown


def verdict(score: int, findings: list[Finding]) -> str:
    if any(f.severity is Severity.CRITICAL for f in findings):
        return "DO NOT TRAIN FINAL MODEL YET"
    if score >= 90:
        return "READY TO TRAIN"
    return "TRAIN WITH CAUTION - REVIEW WARNINGS"
