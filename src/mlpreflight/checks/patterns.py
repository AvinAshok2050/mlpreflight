"""Target Patterns - describes HOW the top features relate to the target,
in plain directional language ("churners spend 28% less"), rather than just
flagging suspicious statistical strength like leakage.py does.

This is deliberately informational, not risk-scoring: it never sets severity
above PASSED and never affects the readiness score. Its only job is to give
the user a fast, human-readable feel for the data before they start modeling.
"""
from __future__ import annotations

import pandas as pd

from ..core.context import InspectionContext
from ..core.finding import ColumnEvidence, Finding, Severity
from ..core.registry import register
from .leakage import SUSPICIOUS_ASSOC, _association_with_target

CATEGORY = "Target Patterns"
TOP_N = 5


def _describe_numeric(ctx: InspectionContext, col: str) -> str | None:
    """'-> 28% lower for class 1' style summary for a numeric feature."""
    y = ctx.y_train
    classes = y.dropna().unique()
    if len(classes) != 2:
        return None
    c0, c1 = sorted(classes, key=str)
    m0 = ctx.train.loc[y == c0, col].mean()
    m1 = ctx.train.loc[y == c1, col].mean()
    if pd.isna(m0) or pd.isna(m1) or m0 == 0:
        return None
    pct = (m1 - m0) / abs(m0) * 100
    direction = "higher" if pct > 0 else "lower"
    return f"class {c1} averages {abs(pct):.0f}% {direction} `{col}` than class {c0}"


def _describe_categorical(ctx: InspectionContext, col: str) -> str | None:
    """'-> class 1 rate is 3x higher for value X' style summary."""
    y = ctx.y_train
    classes = y.dropna().unique()
    if len(classes) != 2:
        return None
    c0, c1 = sorted(classes, key=str)
    rates = ctx.train.groupby(col, observed=True)[ctx.target].apply(
        lambda s: (s == c1).mean()
    )
    if rates.empty or rates.max() == 0:
        return None
    top_cat = rates.idxmax()
    overall_rate = (y == c1).mean()
    if overall_rate == 0:
        return None
    lift = rates.loc[top_cat] / overall_rate
    return (f"`{col}` = {top_cat!r} sees class {c1} at {lift:.1f}x the overall rate "
            f"({rates.loc[top_cat]:.1%} vs {overall_rate:.1%} overall)")


@register("patterns.target_patterns")
def check_target_patterns(ctx: InspectionContext) -> list[Finding]:
    if ctx.y_train.nunique(dropna=True) != 2:
        return [Finding(
            check_id="patterns.target_patterns", category=CATEGORY,
            title="Target patterns", severity=Severity.SKIPPED,
            what="Directional pattern summaries currently support binary targets only.",
            why="This check could not run, so no assurance is provided either way.",
            recommendation="No action needed for multi-class targets.",
            weight=0.0,
        )]

    assoc = _association_with_target(ctx)
    # Exclude identifier-like columns (meaningless per-value "patterns") and
    # anything that already exceeds the leakage threshold - a column about to
    # be recommended for REMOVAL doesn't need its "pattern" described, since
    # the pattern would just be an artifact of the leak itself.
    candidates = {c: v for c, v in assoc.items()
                  if c not in ctx.id_like_cols and v < SUSPICIOUS_ASSOC}
    top_cols = sorted(candidates, key=candidates.get, reverse=True)[:TOP_N]

    descriptions: list[str] = []
    ev: list[ColumnEvidence] = []
    for col in top_cols:
        if assoc[col] < 0.02:
            continue
        if col in ctx.numeric_cols:
            desc = _describe_numeric(ctx, col)
        else:
            desc = _describe_categorical(ctx, col)
        if desc:
            descriptions.append(desc)
            ev.append(ColumnEvidence(col, "association_with_target", assoc[col]))

    if not descriptions:
        return [Finding(
            check_id="patterns.target_patterns", category=CATEGORY,
            title="No strong directional patterns found", severity=Severity.PASSED,
            what="No feature showed a clear directional relationship with the target.",
            why="Useful to know before modeling - the signal may be weak or spread "
                "across many weak features rather than concentrated in a few.",
            recommendation="No action needed.",
            weight=0.0,
        )]

    return [Finding(
        check_id="patterns.target_patterns", category=CATEGORY,
        title="Target patterns", severity=Severity.PASSED,
        what="; ".join(descriptions),
        why="Understanding these relationships up front helps you sanity-check "
            "whether the model's eventual behaviour matches real-world intuition.",
        recommendation="No action needed - informational only.",
        columns=ev,
        weight=0.0,   # never affects the readiness score
    )]