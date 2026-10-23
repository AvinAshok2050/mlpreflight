"""Console rendering."""
from __future__ import annotations

import textwrap

from ..core.finding import Severity

W = 62


def _row(text: str = "") -> str:
    return "| " + text.ljust(W - 4) + " |"


def _rule(ch: str = "-") -> str:
    return "+" + ch * (W - 2) + "+"


def _kv(k: str, v: str) -> str:
    return _row(f"{k:<20}{v}")


def render_summary(report) -> str:
    m = report.meta
    lines = [_rule("=")]
    lines.append(_row("MLPREFLIGHT REPORT".center(W - 4)))
    lines.append(_rule("="))
    lines.append(_kv("Dataset", f"{m.get('n_rows', 0):,} x {m.get('n_cols', 0)}"))
    lines.append(_kv("MODEL READINESS", f"{report.score} / 100"))
    lines.append(_row())
    lines.append(_kv("Critical", str(len(report.critical))))
    lines.append(_kv("Warnings", str(len(report.warnings))))
    lines.append(_kv("Passed", str(len(report.passed))))
    if report.skipped:
        lines.append(_kv("Skipped", f"{len(report.skipped)}  (not assessed)"))

    # --- TARGET ---
    lines.append(_rule())
    lines.append(_row("TARGET"))
    lines.append(_row())
    lines.append(_kv("Target", str(m.get("target", "-"))))
    n_classes = m.get("n_classes")
    lines.append(_kv("Classes", str(n_classes)))
    balance = m.get("class_balance") or {}
    for cls, pct in balance.items():
        lines.append(_kv(f"  class {cls}", f"{pct:.1%}"))

    # --- DATA QUALITY / FEATURE RISK (critical + warning findings, grouped) ---
    for section_title, categories in [
        ("DATA QUALITY", {"Dataset & Target Health"}),
        ("FEATURE RISK", {"Feature Risk", "Leakage & Split Integrity"}),
        ("TRAIN/TEST COMPATIBILITY", {"Train/Test Compatibility"}),
        ("CLASSIFICATION RISK", {"Classification Risk"}),
    ]:
        group = [f for f in (report.critical + report.warnings) if f.category in categories]
        if not group:
            continue
        lines.append(_rule())
        lines.append(_row(section_title))
        lines.append(_row())
        for f in group:
            lines.append(_row(f"{f.severity.icon} {f.title}"))
            cols = ", ".join(f.affected_columns[:4])
            if cols and cols != "__rows__":
                for chunk in textwrap.wrap(f"   -> {cols}", W - 6):
                    lines.append(_row(chunk))
            for chunk in textwrap.wrap(f"   {f.recommendation}", W - 6):
                lines.append(_row(chunk))

    # --- TARGET PATTERNS ---
    patterns = report.target_patterns()
    if patterns:
        lines.append(_rule())
        lines.append(_row("TARGET PATTERNS"))
        lines.append(_row())
        for p in patterns:
            for chunk in textwrap.wrap(f"- {p}", W - 6):
                lines.append(_row(chunk))

    # --- RECOMMENDED FEATURES ---
    rec = report.recommended_features()
    lines.append(_rule())
    lines.append(_row("RECOMMENDED FEATURES"))
    lines.append(_row())
    lines.append(_kv("KEEP", f"{len(rec['keep'])} features"))
    if rec["investigate"]:
        lines.append(_kv("INVESTIGATE", ", ".join(rec["investigate"][:6])))
    if rec["remove"]:
        lines.append(_kv("REMOVE", ", ".join(rec["remove"][:6])))

    if report.assumptions:
        lines.append(_rule())
        lines.append(_row("ASSUMPTIONS MADE"))
        lines.append(_row())
        for a in report.assumptions:
            for chunk in textwrap.wrap(a, W - 6):
                lines.append(_row(chunk))

    lines.append(_rule())
    lines.append(_row("RECOMMENDATION"))
    lines.append(_row())
    lines.append(_row(report.verdict))
    lines.append(_rule("="))
    return "\n".join(lines)


def render_detail(report, severity: str | None = None) -> str:
    order = [Severity.CRITICAL, Severity.WARNING, Severity.PASSED, Severity.SKIPPED]
    if severity:
        order = [Severity(severity)]

    out: list[str] = []
    for sev in order:
        group = [f for f in report.findings if f.severity is sev]
        if not group:
            continue
        out.append(f"\n{'=' * W}\n{sev.value.upper()}  ({len(group)})\n{'=' * W}")
        for f in group:
            out.append(f"\n{f.severity.icon} {f.title}   [{f.check_id}]")
            out.append(f"   Category   : {f.category}")
            out.append("   What       : " + _wrap(f.what))
            if sev in (Severity.CRITICAL, Severity.WARNING):
                out.append("   Why        : " + _wrap(f.why))
                out.append("   Action     : " + _wrap(f.recommendation))
                if f.confidence < 1.0:
                    out.append(f"   Confidence : {f.confidence:.0%} (assumption-dependent)")
                if f.columns:
                    out.append("   Columns    :")
                    for ev in f.columns[:10]:
                        thr = f" (threshold {ev.threshold})" if ev.threshold is not None else ""
                        out.append(f"      - {ev.column}: {ev.statistic}={ev.value:.4f}{thr}")
                    if len(f.columns) > 10:
                        out.append(f"      ... and {len(f.columns) - 10} more")
    return "\n".join(out)


def _wrap(text: str, indent: int = 16) -> str:
    wrapped = textwrap.wrap(text, W + 20)
    if not wrapped:
        return ""
    return ("\n" + " " * indent).join(wrapped)