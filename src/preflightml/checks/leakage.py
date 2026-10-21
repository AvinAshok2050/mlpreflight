"""Leakage & Split Integrity checks.

This module is the library's primary differentiator. Each check returns
column-level evidence so a user can drill from "leakage detected" straight to
the offending fields and the statistic that triggered the flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.context import InspectionContext
from ..core.finding import ColumnEvidence, Finding, Severity
from ..core.registry import register

CATEGORY = "Leakage & Split Integrity"

# A feature explaining nearly all target variance is almost always leakage
# rather than a genuinely excellent predictor.
SUSPICIOUS_ASSOC = 0.95
ELEVATED_ASSOC = 0.80


def _association_with_target(ctx: InspectionContext) -> dict[str, float]:
    """Absolute rank (Spearman) correlation of every feature with the target.

    Computed manually from ranks so it needs only pandas/numpy. (pandas'
    own method="spearman" silently requires scipy, and phik was removed
    because it becomes extremely slow on high-cardinality columns such as
    IDs and timestamps.)

    Non-numeric columns are encoded as integer category codes first, so for
    nominal categoricals this is an approximation, not a true categorical
    association measure.
    """
    out: dict[str, float] = {}
    y = pd.Series(pd.factorize(ctx.y_train)[0])
    y_rank = y.rank()
    for col in ctx.features:
        s = ctx.train[col]
        try:
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                x = s.reset_index(drop=True)
            else:
                x = pd.Series(pd.factorize(s)[0])
            out[col] = abs(float(x.rank().corr(y_rank)))
        except Exception:
            out[col] = float("nan")
    return {k: (0.0 if np.isnan(v) else v) for k, v in out.items()}


@register("leakage.target_association")
def check_target_leakage(ctx: InspectionContext) -> list[Finding]:
    assoc = _association_with_target(ctx)
    critical = [(c, v) for c, v in assoc.items() if v >= SUSPICIOUS_ASSOC]
    elevated = [(c, v) for c, v in assoc.items() if ELEVATED_ASSOC <= v < SUSPICIOUS_ASSOC]

    if critical:
        ev = [ColumnEvidence(c, "association_with_target", v, SUSPICIOUS_ASSOC)
              for c, v in sorted(critical, key=lambda t: -t[1])]
        names = ", ".join(f"`{c}`" for c, _ in critical[:5])
        return [Finding(
            check_id="leakage.target_association",
            category=CATEGORY,
            title="Potential target leakage",
            severity=Severity.CRITICAL,
            what=f"{len(critical)} feature(s) are almost perfectly associated with "
                 f"the target: {names}.",
            why="A feature this predictive is usually derived from the outcome or "
                "recorded after it. The model will score well in validation and "
                "fail in production, where the feature is unavailable or unfilled.",
            recommendation="Inspect how each column is populated and when it becomes "
                           "available relative to the prediction point. Drop any "
                           "column not knowable at prediction time.",
            columns=ev,
            weight=1.5,
        )]

    if elevated:
        ev = [ColumnEvidence(c, "association_with_target", v, ELEVATED_ASSOC)
              for c, v in sorted(elevated, key=lambda t: -t[1])]
        names = ", ".join(f"`{c}`" for c, _ in elevated[:5])
        return [Finding(
            check_id="leakage.target_association",
            category=CATEGORY,
            title="Unusually strong feature-target relationship",
            severity=Severity.WARNING,
            what=f"{len(elevated)} feature(s) show strong association with the "
                 f"target: {names}.",
            why="This may be a genuinely strong signal, or an early sign of leakage. "
                "Worth confirming before it silently inflates your metrics.",
            recommendation="Confirm each column is populated before the outcome occurs.",
            columns=ev,
            confidence=0.7,
        )]

    return [Finding(
        check_id="leakage.target_association",
        category=CATEGORY,
        title="No suspicious feature-target relationships",
        severity=Severity.PASSED,
        what="No feature exceeded the leakage association threshold.",
        why="Reduces the risk of metrics that will not survive production.",
        recommendation="No action needed.",
    )]


@register("leakage.identifier_columns")
def check_identifier_leakage(ctx: InspectionContext) -> list[Finding]:
    if not ctx.id_like_cols:
        return [Finding(
            check_id="leakage.identifier_columns", category=CATEGORY,
            title="No identifier-like features", severity=Severity.PASSED,
            what="No near-unique or identifier-named columns found among features.",
            why="Identifiers let tree models memorise rows instead of generalising.",
            recommendation="No action needed.",
        )]

    ev = []
    for c in ctx.id_like_cols:
        ratio = ctx.train[c].nunique(dropna=True) / max(len(ctx.train), 1)
        ev.append(ColumnEvidence(c, "unique_ratio", ratio, 0.95))
    names = ", ".join(f"`{c}`" for c in ctx.id_like_cols[:5])

    return [Finding(
        check_id="leakage.identifier_columns", category=CATEGORY,
        title="Identifier-like columns used as features",
        severity=Severity.WARNING,
        what=f"{len(ctx.id_like_cols)} near-unique or identifier-named column(s) "
             f"are present as features: {names}.",
        why="Identifiers carry no generalisable signal but can be memorised, "
            "inflating validation scores. They also frequently encode ordering "
            "or cohort information that leaks the target.",
        recommendation="Drop these columns, or keep them only as grouping keys for "
                       "splitting rather than as model inputs.",
        columns=ev,
    )]


@register("leakage.row_overlap")
def check_row_overlap(ctx: InspectionContext) -> list[Finding]:
    if not ctx.has_test:
        return [_skipped("leakage.row_overlap", "Train/test row overlap",
                         "No test frame was supplied.",
                         "Pass test= to enable split integrity checks.")]

    cols = [c for c in ctx.features if c in ctx.test.columns]
    # Hash each row's values. Unlike joining rows into strings, this handles
    # missing values correctly on every pandas version (on pandas 3.x,
    # astype(str) keeps NaN as a float, which breaks "|".join).
    tr = pd.util.hash_pandas_object(ctx.train[cols], index=False)
    te = pd.util.hash_pandas_object(ctx.test[cols], index=False)
    overlap = len(set(tr) & set(te))
    pct = overlap / max(len(te), 1)

    if overlap == 0:
        return [Finding(
            check_id="leakage.row_overlap", category=CATEGORY,
            title="No train/test row overlap", severity=Severity.PASSED,
            what="No identical feature rows appear in both train and test.",
            why="Overlap inflates test metrics by scoring the model on rows it "
                "already memorised.",
            recommendation="No action needed.",
        )]

    sev = Severity.CRITICAL if pct > 0.01 else Severity.WARNING
    return [Finding(
        check_id="leakage.row_overlap", category=CATEGORY,
        title="Train/test row overlap", severity=sev,
        what=f"{overlap:,} test rows ({pct:.1%}) appear identically in the training set.",
        why="Test metrics measured on memorised rows are not an estimate of "
            "generalisation. Reported performance will be optimistic.",
        recommendation="Deduplicate before splitting, and confirm the split was not "
                       "applied after an operation that duplicated rows.",
        columns=[ColumnEvidence("__rows__", "overlap_fraction", pct, 0.01,
                                {"overlapping_rows": overlap})],
        weight=1.5,
    )]


@register("leakage.entity_overlap")
def check_entity_overlap(ctx: InspectionContext) -> list[Finding]:
    if not ctx.has_test or ctx.entity_col is None:
        return [_skipped("leakage.entity_overlap", "Entity overlap across split",
                         "No test frame or no entity column identified.",
                         "Pass test= and entity_col= to enable this check.")]
    col = ctx.entity_col
    if col not in ctx.test.columns:
        return [_skipped("leakage.entity_overlap", "Entity overlap across split",
                         f"Column {col!r} missing from test frame.",
                         "Ensure both frames share the entity column.")]

    tr_e, te_e = set(ctx.train[col].dropna()), set(ctx.test[col].dropna())
    shared = tr_e & te_e
    pct = len(shared) / max(len(te_e), 1)
    conf = 0.6 if ctx.inferred("entity_col") else 1.0

    if not shared:
        return [Finding(
            check_id="leakage.entity_overlap", category=CATEGORY,
            title="No entity overlap across split", severity=Severity.PASSED,
            what=f"No values of `{col}` appear in both train and test.",
            why="Entity overlap lets the model learn per-entity quirks rather than "
                "generalisable patterns.",
            recommendation="No action needed.",
            confidence=conf, assumptions=list(ctx.assumptions) if conf < 1 else [],
        )]

    return [Finding(
        check_id="leakage.entity_overlap", category=CATEGORY,
        title="Entity overlap between train and test",
        severity=Severity.CRITICAL if pct > 0.05 else Severity.WARNING,
        what=f"{len(shared):,} values of `{col}` ({pct:.1%} of test entities) appear "
             f"in both train and test.",
        why="The model can memorise entity-specific behaviour seen in training and "
            "be rewarded for it at test time. Performance on genuinely new entities "
            "will be substantially worse than reported.",
        recommendation=f"Use a grouped split (e.g. GroupKFold / GroupShuffleSplit) "
                       f"keyed on `{col}` so each entity appears in only one fold.",
        columns=[ColumnEvidence(col, "entity_overlap_fraction", pct, 0.05,
                                {"shared_entities": len(shared)})],
        confidence=conf, weight=1.5,
        assumptions=list(ctx.assumptions) if conf < 1 else [],
    )]


@register("leakage.temporal")
def check_temporal_leakage(ctx: InspectionContext) -> list[Finding]:
    if not ctx.has_test or ctx.time_col is None:
        return [_skipped("leakage.temporal", "Temporal leakage",
                         "No test frame or no time column identified.",
                         "Pass test= and time_col= to enable this check.")]
    col = ctx.time_col
    if col not in ctx.test.columns:
        return [_skipped("leakage.temporal", "Temporal leakage",
                         f"Column {col!r} missing from test frame.",
                         "Ensure both frames share the time column.")]

    tr = pd.to_datetime(ctx.train[col], errors="coerce").dropna()
    te = pd.to_datetime(ctx.test[col], errors="coerce").dropna()
    if tr.empty or te.empty:
        return [_skipped("leakage.temporal", "Temporal leakage",
                         f"Column {col!r} could not be parsed as datetime.",
                         "Convert the column to datetime before inspecting.")]

    n_after = int((tr > te.min()).sum())
    pct = n_after / max(len(tr), 1)
    conf = 0.6 if ctx.inferred("time_col") else 1.0

    if n_after == 0:
        return [Finding(
            check_id="leakage.temporal", category=CATEGORY,
            title="Split respects time ordering", severity=Severity.PASSED,
            what=f"All training rows precede the earliest test row on `{col}`.",
            why="A time-respecting split is required for an honest estimate when the "
                "model will predict forward in time.",
            recommendation="No action needed.", confidence=conf,
        )]

    return [Finding(
        check_id="leakage.temporal", category=CATEGORY,
        title="Temporal leakage",
        severity=Severity.CRITICAL if pct > 0.05 else Severity.WARNING,
        what=f"{n_after:,} training rows ({pct:.1%}) occur after the earliest test "
             f"row on `{col}` (train max {tr.max()}, test min {te.min()}).",
        why="The model is trained on information from the future relative to part of "
            "the test set. This is the single most common cause of a model that "
            "validates well and then degrades immediately after deployment.",
        recommendation=f"Split chronologically on `{col}`: train strictly before a "
                       f"cutoff, test strictly after it.",
        columns=[ColumnEvidence(col, "fraction_train_after_test_start", pct, 0.05,
                                {"train_max": str(tr.max()), "test_min": str(te.min())})],
        confidence=conf, weight=1.5,
        assumptions=list(ctx.assumptions) if conf < 1 else [],
    )]


def _skipped(check_id: str, title: str, what: str, rec: str) -> Finding:
    return Finding(
        check_id=check_id, category=CATEGORY, title=title,
        severity=Severity.SKIPPED, what=what,
        why="This check could not run, so no assurance is provided either way.",
        recommendation=rec,
    )