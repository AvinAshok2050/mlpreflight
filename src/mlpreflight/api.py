"""Public entry point."""
from __future__ import annotations

import pandas as pd

from . import checks  # noqa: F401  ensures checks register themselves
from .core.context import InspectionContext
from .core.registry import all_checks
from .report import Report


def inspect(
    train: pd.DataFrame,
    target: str,
    test: pd.DataFrame | None = None,
    time_col: str | None = None,
    entity_col: str | None = None,
) -> Report:
    """Assess whether a classification dataset is ready to model.

    Parameters
    ----------
    train : the training frame, including the target column.
    target : name of the target column.
    test : optional held-out frame. Required for split-integrity and
        train/test compatibility checks; without it those report as SKIPPED
        rather than PASSED.
    time_col : column defining event time. Enables temporal leakage checks.
        Auto-detected when unambiguous, but passing it explicitly raises the
        confidence of any resulting finding.
    entity_col : column identifying the entity (customer, user, account).
        Enables entity-overlap checks. Same auto-detection caveat applies.

    Returns
    -------
    Report
    """
    ctx = InspectionContext(train=train, target=target, test=test,
                            time_col=time_col, entity_col=entity_col)

    findings = []
    for check_id, fn in all_checks():
        try:
            findings.extend(fn(ctx))
        except Exception as exc:  # a broken check must never break the run
            from .core.finding import Finding, Severity
            findings.append(Finding(
                check_id=check_id, category="Internal", title=f"Check failed: {check_id}",
                severity=Severity.SKIPPED, what=f"{type(exc).__name__}: {exc}",
                why="This check could not complete, so no assurance is provided.",
                recommendation="Please report this at the project issue tracker."))

    meta = {
        "n_rows": len(train), "n_cols": train.shape[1], "target": target,
        "n_classes": int(train[target].nunique(dropna=True)),
        "has_test": test is not None,
        "n_test_rows": len(test) if test is not None else 0,
        "time_col": ctx.time_col, "entity_col": ctx.entity_col,
        "features": ctx.features,
        "class_balance": (train[target].value_counts(normalize=True).to_dict()
                          if train[target].nunique(dropna=True) <= 10 else {}),
    }
    return Report(findings=findings, meta=meta, assumptions=ctx.assumptions)