"""Core data model for findings produced by checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Outcome of a single check.

    SKIPPED is deliberately distinct from PASSED: a check that could not run
    must never be presented as a check that succeeded.
    """

    CRITICAL = "critical"
    WARNING = "warning"
    PASSED = "passed"
    SKIPPED = "skipped"

    @property
    def icon(self) -> str:
        return {
            "critical": "[!]",
            "warning": "[~]",
            "passed": "[+]",
            "skipped": "[-]",
        }[self.value]

    @property
    def rank(self) -> int:
        return {"critical": 0, "warning": 1, "passed": 2, "skipped": 3}[self.value]


@dataclass
class ColumnEvidence:
    """Per-column evidence backing a finding.

    This is the heart of the library's differentiation: every finding must be
    traceable to specific columns with the statistics that triggered it.
    """

    column: str
    statistic: str          # e.g. "phik_vs_target", "psi", "cardinality_ratio"
    value: float
    threshold: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "statistic": self.statistic,
            "value": round(float(self.value), 6),
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass
class Finding:
    """A single check result."""

    check_id: str                       # stable machine id, e.g. "leakage.target_correlation"
    category: str                       # e.g. "Leakage & Split Integrity"
    title: str                          # human title
    severity: Severity
    what: str                           # what was found
    why: str                            # why it matters
    recommendation: str                 # what to do about it
    columns: list[ColumnEvidence] = field(default_factory=list)
    confidence: float = 1.0             # 0-1; lowered when assumptions were inferred
    assumptions: list[str] = field(default_factory=list)
    weight: float = 1.0                 # contribution to readiness score

    @property
    def affected_columns(self) -> list[str]:
        return [c.column for c in self.columns]

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "category": self.category,
            "title": self.title,
            "severity": self.severity.value,
            "what": self.what,
            "why": self.why,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 3),
            "assumptions": self.assumptions,
            "affected_columns": self.affected_columns,
            "evidence": [c.as_dict() for c in self.columns],
        }
