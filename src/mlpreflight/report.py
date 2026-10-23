"""The Report object returned by inspect()."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .core.finding import Finding, Severity
from .core.registry import compute_score, verdict
from .render.console import render_summary, render_detail
from .render.html import render_html


@dataclass
class Report:
    findings: list[Finding]
    meta: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- #
    @property
    def score(self) -> int:
        return compute_score(self.findings)[0]

    @property
    def verdict(self) -> str:
        return verdict(self.score, self.findings)

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.CRITICAL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def passed(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.PASSED]

    @property
    def skipped(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.SKIPPED]

    @property
    def has_blocking_issues(self) -> bool:
        """CI/CD gate: True if any Critical finding exists."""
        return bool(self.critical)

    # ---------------------------------------------------------------- #
    def summary(self) -> None:
        print(render_summary(self))

    def details(self, severity: str | None = None) -> None:
        print(render_detail(self, severity))

    def score_breakdown(self) -> list[dict]:
        """Every point deducted, so the score can be reconstructed by hand."""
        return compute_score(self.findings)[1]

    def column_report(self) -> dict[str, list[dict]]:
        """Invert findings into a column -> issues map for drill-down."""
        out: dict[str, list[dict]] = {}
        for f in self.findings:
            if f.severity in (Severity.PASSED, Severity.SKIPPED):
                continue
            for ev in f.columns:
                out.setdefault(ev.column, []).append({
                    "check_id": f.check_id,
                    "title": f.title,
                    "severity": f.severity.value,
                    "statistic": ev.statistic,
                    "value": ev.value,
                    "threshold": ev.threshold,
                    "detail": ev.detail,
                })
        return out

    def recommended_features(self) -> dict[str, list[str]]:
        """Aggregate every finding into a single KEEP / INVESTIGATE / REMOVE
        verdict per feature - the actionable summary a user can act on
        directly, instead of reading every individual finding by hand.

        Rule: a column touched by any CRITICAL finding -> REMOVE.
              a column touched only by WARNING findings -> INVESTIGATE.
              every other feature (not the target, no risk findings) -> KEEP.
        """
        remove: set[str] = set()
        investigate: set[str] = set()

        for f in self.findings:
            if f.severity is Severity.CRITICAL:
                remove.update(c for c in f.affected_columns if not c.startswith("__"))
            elif f.severity is Severity.WARNING:
                investigate.update(c for c in f.affected_columns if not c.startswith("__"))

        investigate -= remove  # a column that's both is a REMOVE, not a maybe

        all_features = set(self.meta.get("features", []))
        keep = all_features - remove - investigate

        return {
            "keep": sorted(keep),
            "investigate": sorted(investigate),
            "remove": sorted(remove),
        }

    def target_patterns(self) -> list[str]:
        """Plain-language directional relationships found between the top
        features and the target (e.g. 'churners spend 28% less')."""
        for f in self.findings:
            if f.check_id == "patterns.target_patterns" and f.severity is Severity.PASSED:
                return [s.strip() for s in f.what.split(";")] if ";" in f.what else (
                    [] if f.what.startswith("No ") else [f.what]
                )
        return []

    def explain(self, column: str) -> None:
        """Drill down into a single column."""
        issues = self.column_report().get(column)
        if not issues:
            print(f"No issues recorded for column `{column}`.")
            return
        print(f"\nColumn: {column}\n" + "-" * 60)
        for i in issues:
            print(f"  [{i['severity'].upper()}] {i['title']}")
            print(f"      {i['statistic']} = {i['value']:.4f}"
                  + (f" (threshold {i['threshold']})" if i["threshold"] is not None else ""))
            for k, v in i["detail"].items():
                print(f"      {k}: {v}")

    # ---------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "score": self.score,
            "verdict": self.verdict,
            "counts": {
                "critical": len(self.critical),
                "warning": len(self.warnings),
                "passed": len(self.passed),
                "skipped": len(self.skipped),
            },
            "assumptions": self.assumptions,
            "score_breakdown": self.score_breakdown(),
            "findings": [f.as_dict() for f in self.findings],
            "columns": self.column_report(),
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            with open(path, "w") as fh:
                fh.write(payload)
        return payload

    def to_html(self, path: str | None = None) -> str:
        html = render_html(self)
        if path:
            with open(path, "w") as fh:
                fh.write(html)
        return html

    def __repr__(self) -> str:
        return (f"<Report score={self.score}/100 critical={len(self.critical)} "
                f"warnings={len(self.warnings)} passed={len(self.passed)} "
                f"skipped={len(self.skipped)}>")