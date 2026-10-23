"""Dataset Health, Train/Test Compatibility, Feature Risk, Classification Risk."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.context import InspectionContext
from ..core.finding import ColumnEvidence, Finding, Severity
from ..core.registry import register

HEALTH = "Dataset & Target Health"
COMPAT = "Train/Test Compatibility"
FEATURE = "Feature Risk"
CLASSIF = "Classification Risk"


# --------------------------------------------------------------------- #
# Dataset & Target Health
# --------------------------------------------------------------------- #
@register("health.missing_values")
def check_missing(ctx: InspectionContext) -> list[Finding]:
    frac = ctx.train[ctx.features].isna().mean()
    bad = frac[frac > 0.4].sort_values(ascending=False)
    if bad.empty:
        return [_ok("health.missing_values", HEALTH, "Missing values within tolerance",
                    "No feature exceeds 40% missing values.",
                    "Heavily missing columns rarely carry usable signal.")]
    ev = [ColumnEvidence(c, "missing_fraction", float(v), 0.4) for c, v in bad.items()]
    return [Finding(
        check_id="health.missing_values", category=HEALTH,
        title="Columns with heavy missingness",
        severity=Severity.CRITICAL if (bad > 0.8).any() else Severity.WARNING,
        what=f"{len(bad)} column(s) exceed 40% missing: "
             + ", ".join(f"`{c}` ({v:.0%})" for c, v in bad.head(5).items()),
        why="Imputing a mostly-empty column invents most of its values. If the "
            "missingness itself relates to the target, imputation can also destroy "
            "real signal or introduce leakage.",
        recommendation="Drop these columns, or replace them with an explicit "
                       "is_missing indicator if the missingness is informative.",
        columns=ev)]


@register("health.duplicate_rows")
def check_duplicates(ctx: InspectionContext) -> list[Finding]:
    n_dup = int(ctx.train.duplicated().sum())
    pct = n_dup / max(len(ctx.train), 1)
    if n_dup == 0:
        return [_ok("health.duplicate_rows", HEALTH, "No duplicate rows",
                    "No fully duplicated rows in the training set.",
                    "Duplicates distort class balance and can cross the split.")]
    return [Finding(
        check_id="health.duplicate_rows", category=HEALTH, title="Duplicate rows",
        severity=Severity.CRITICAL if pct > 0.05 else Severity.WARNING,
        what=f"{n_dup:,} duplicated rows ({pct:.1%} of the training set).",
        why="Duplicates silently reweight the data and, if introduced before "
            "splitting, place identical rows on both sides of the split.",
        recommendation="Deduplicate before splitting, and confirm whether the "
                       "duplication is a genuine data issue or an upstream join bug.",
        columns=[ColumnEvidence("__rows__", "duplicate_fraction", pct, 0.05,
                                {"duplicate_rows": n_dup})])]


@register("health.constant_columns")
def check_constant(ctx: InspectionContext) -> list[Finding]:
    ev = []
    for c in ctx.features:
        vc = ctx.train[c].value_counts(normalize=True, dropna=False)
        if not vc.empty and vc.iloc[0] > 0.99:
            ev.append(ColumnEvidence(c, "dominant_value_fraction", float(vc.iloc[0]), 0.99,
                                     {"dominant_value": str(vc.index[0])}))
    if not ev:
        return [_ok("health.constant_columns", HEALTH, "No constant columns",
                    "No feature is dominated by a single value.",
                    "Constant columns add cost without signal.")]
    return [Finding(
        check_id="health.constant_columns", category=HEALTH,
        title="Constant / near-constant columns", severity=Severity.WARNING,
        what=f"{len(ev)} column(s) take one value in >99% of rows: "
             + ", ".join(f"`{e.column}`" for e in ev[:5]),
        why="These carry effectively no information, inflate encoding width, and can "
            "produce unstable splits in tree models.",
        recommendation="Drop these columns unless the rare value is itself meaningful.",
        columns=ev)]


@register("health.target_quality")
def check_target_quality(ctx: InspectionContext) -> list[Finding]:
    y = ctx.y_train
    n_missing = int(y.isna().sum())
    n_classes = int(y.nunique(dropna=True))
    if n_missing:
        return [Finding(
            check_id="health.target_quality", category=HEALTH,
            title="Target column has missing values", severity=Severity.CRITICAL,
            what=f"`{ctx.target}` is missing in {n_missing:,} rows ({n_missing/len(y):.1%}).",
            why="Rows without a label cannot be trained on. Silently dropping them can "
                "also bias the sample if the missingness is not random.",
            recommendation="Investigate why labels are missing before dropping the rows.",
            columns=[ColumnEvidence(ctx.target, "missing_count", n_missing, 0)],
            weight=1.5)]
    if n_classes < 2:
        return [Finding(
            check_id="health.target_quality", category=HEALTH,
            title="Target has fewer than two classes", severity=Severity.CRITICAL,
            what=f"`{ctx.target}` contains {n_classes} distinct value(s).",
            why="A classification problem requires at least two classes.",
            recommendation="Check the target definition and the filtering applied upstream.",
            columns=[ColumnEvidence(ctx.target, "n_classes", n_classes, 2)], weight=1.5)]
    return [_ok("health.target_quality", HEALTH, "Target column is well-formed",
                f"`{ctx.target}` has {n_classes} classes and no missing values.",
                "A clean target is a precondition for every downstream check.")]


@register("health.sample_size")
def check_sample_size(ctx: InspectionContext) -> list[Finding]:
    n, p = len(ctx.train), len(ctx.features)
    ratio = n / max(p, 1)
    if ratio >= 50:
        return [_ok("health.sample_size", HEALTH, "Adequate rows per feature",
                    f"{n:,} rows for {p} features ({ratio:.0f}:1).",
                    "Thin data relative to feature count invites overfitting.")]
    return [Finding(
        check_id="health.sample_size", category=HEALTH,
        title="Low row-to-feature ratio",
        severity=Severity.CRITICAL if ratio < 10 else Severity.WARNING,
        what=f"{n:,} rows for {p} features — a ratio of {ratio:.1f}:1.",
        why="With few rows per feature, a flexible model can fit noise well enough to "
            "produce a validation score that will not reproduce on new data.",
        recommendation="Reduce dimensionality, apply stronger regularisation, and "
                       "prefer repeated cross-validation over a single holdout.",
        columns=[ColumnEvidence("__dataset__", "rows_per_feature", ratio, 50.0)])]


# --------------------------------------------------------------------- #
# Train/Test Compatibility
# --------------------------------------------------------------------- #
def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions."""
    e, a = expected.dropna(), actual.dropna()
    if e.empty or a.empty:
        return 0.0
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e_pct = np.histogram(e, bins=edges)[0] / len(e)
    a_pct = np.histogram(a, bins=edges)[0] / len(a)
    e_pct, a_pct = np.clip(e_pct, 1e-6, None), np.clip(a_pct, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


@register("compat.numeric_shift")
def check_numeric_shift(ctx: InspectionContext) -> list[Finding]:
    if not ctx.has_test:
        return [_skip("compat.numeric_shift", COMPAT, "Numeric distribution shift")]
    ev = []
    for c in ctx.numeric_cols:
        if c not in ctx.test.columns:
            continue
        v = _psi(ctx.train[c], ctx.test[c])
        if v > 0.1:
            ev.append(ColumnEvidence(c, "psi", v, 0.1))
    if not ev:
        return [_ok("compat.numeric_shift", COMPAT, "Numeric distributions are stable",
                    "No numeric feature shows meaningful train/test drift (PSI <= 0.1).",
                    "Drift between splits means the test score measures a different "
                    "population than the model was trained on.")]
    ev.sort(key=lambda e: -e.value)
    severe = [e for e in ev if e.value > 0.25]
    return [Finding(
        check_id="compat.numeric_shift", category=COMPAT,
        title="Train/test distribution shift",
        severity=Severity.CRITICAL if severe else Severity.WARNING,
        what=f"{len(ev)} numeric feature(s) differ between train and test: "
             + ", ".join(f"`{e.column}` (PSI {e.value:.2f})" for e in ev[:5]),
        why="The test set represents a different population than the training set, so "
            "the test score is not a clean estimate of performance on the data the "
            "model was fitted to. This often signals a split made along a hidden "
            "time or cohort boundary.",
        recommendation="Check how the split was produced. If the shift is genuine and "
                       "expected in production, keep it but report it explicitly.",
        columns=ev)]


@register("compat.new_categories")
def check_new_categories(ctx: InspectionContext) -> list[Finding]:
    if not ctx.has_test:
        return [_skip("compat.new_categories", COMPAT, "Unseen categories in test")]
    ev = []
    for c in ctx.categorical_cols:
        if c not in ctx.test.columns:
            continue
        tr, te = set(ctx.train[c].dropna()), set(ctx.test[c].dropna())
        new = te - tr
        if new:
            frac = ctx.test[c].isin(new).mean()
            ev.append(ColumnEvidence(c, "unseen_category_fraction", float(frac), 0.0,
                                     {"n_unseen": len(new),
                                      "examples": [str(x) for x in list(new)[:5]]}))
    if not ev:
        return [_ok("compat.new_categories", COMPAT, "No unseen categories in test",
                    "Every categorical value in test also appears in train.",
                    "Unseen categories break encoders at inference time.")]
    ev.sort(key=lambda e: -e.value)
    return [Finding(
        check_id="compat.new_categories", category=COMPAT,
        title="Categories present in test but not train",
        severity=Severity.WARNING,
        what=f"{len(ev)} column(s) contain unseen values in test: "
             + ", ".join(f"`{e.column}` ({e.detail['n_unseen']} new)" for e in ev[:5]),
        why="Encoders fitted on train will fail or silently map these to a default, "
            "quietly degrading predictions on exactly the rows that are most novel.",
        recommendation="Configure encoders with handle_unknown='ignore' (or an explicit "
                       "unknown bucket), and confirm the categories are legitimate.",
        columns=ev)]


@register("compat.target_shift")
def check_target_shift(ctx: InspectionContext) -> list[Finding]:
    if not ctx.has_test or ctx.target not in ctx.test.columns:
        return [_skip("compat.target_shift", COMPAT, "Target rate shift")]
    tr = ctx.y_train.value_counts(normalize=True)
    te = ctx.test[ctx.target].value_counts(normalize=True)
    diff = (tr - te).abs().max()
    if pd.isna(diff) or diff < 0.05:
        return [_ok("compat.target_shift", COMPAT, "Target rate is stable across split",
                    "Class proportions match between train and test.",
                    "A shifted base rate makes test metrics incomparable to train.")]
    return [Finding(
        check_id="compat.target_shift", category=COMPAT, title="Target rate shift",
        severity=Severity.WARNING,
        what=f"Class proportions differ between train and test by up to {diff:.1%}.",
        why="Threshold-dependent metrics and calibrated probabilities assume a stable "
            "base rate. A shift makes the test score measure a different problem.",
        recommendation="Use a stratified split, or if the shift is real, recalibrate "
                       "and report metrics that are robust to base rate.",
        columns=[ColumnEvidence(ctx.target, "max_class_proportion_diff", float(diff), 0.05)])]


# --------------------------------------------------------------------- #
# Feature Risk
# --------------------------------------------------------------------- #
@register("feature.high_cardinality")
def check_cardinality(ctx: InspectionContext) -> list[Finding]:
    ev = []
    for c in ctx.categorical_cols:
        n_uniq = ctx.train[c].nunique(dropna=True)
        if n_uniq > 50 and c not in ctx.id_like_cols:
            ev.append(ColumnEvidence(c, "n_categories", float(n_uniq), 50.0))
    if not ev:
        return [_ok("feature.high_cardinality", FEATURE, "Categorical cardinality is manageable",
                    "No categorical feature exceeds 50 distinct values.",
                    "High cardinality inflates encodings and invites target-encoding leakage.")]
    ev.sort(key=lambda e: -e.value)
    return [Finding(
        check_id="feature.high_cardinality", category=FEATURE,
        title="High-cardinality categorical features", severity=Severity.WARNING,
        what=f"{len(ev)} column(s) have many distinct values: "
             + ", ".join(f"`{e.column}` ({int(e.value)})" for e in ev[:5]),
        why="One-hot encoding explodes the feature space; target encoding on these "
            "columns leaks the target unless fitted strictly inside each CV fold.",
        recommendation="Group rare levels, or use target encoding fitted within folds "
                       "only. Never fit an encoder on the full dataset before splitting.",
        columns=ev)]


@register("feature.redundancy")
def check_redundancy(ctx: InspectionContext) -> list[Finding]:
    num = [c for c in ctx.numeric_cols]
    if len(num) < 2:
        return [_skip("feature.redundancy", FEATURE, "Feature redundancy")]
    corr = ctx.train[num].corr().abs()
    ev = []
    seen = set()
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and v > 0.95 and (a, b) not in seen:
                seen.add((a, b))
                ev.append(ColumnEvidence(b, "abs_correlation", float(v), 0.95,
                                         {"redundant_with": a}))
    if not ev:
        return [_ok("feature.redundancy", FEATURE, "No near-duplicate features",
                    "No numeric feature pair exceeds 0.95 absolute correlation.",
                    "Redundant features destabilise coefficients and split importances.")]
    return [Finding(
        check_id="feature.redundancy", category=FEATURE,
        title="Highly redundant features", severity=Severity.WARNING,
        what=f"{len(ev)} feature pair(s) are near-duplicates: "
             + ", ".join(f"`{e.column}`~`{e.detail['redundant_with']}`" for e in ev[:5]),
        why="Collinear features split importance between themselves, making the model "
            "harder to interpret and coefficient-based explanations unreliable.",
        recommendation="Keep one feature from each redundant pair, or combine them.",
        columns=ev)]


# --------------------------------------------------------------------- #
# Classification Risk
# --------------------------------------------------------------------- #
@register("classification.imbalance")
def check_imbalance(ctx: InspectionContext) -> list[Finding]:
    vc = ctx.y_train.value_counts()
    if len(vc) < 2:
        return [_skip("classification.imbalance", CLASSIF, "Class imbalance")]
    minority, majority = vc.index[-1], vc.index[0]
    n_min, n_maj = int(vc.iloc[-1]), int(vc.iloc[0])
    ratio = n_maj / max(n_min, 1)
    minority_pct = n_min / len(ctx.y_train)

    ev = [ColumnEvidence(ctx.target, "imbalance_ratio", ratio, 3.0,
                         {"minority_class": str(minority), "minority_count": n_min,
                          "minority_pct": round(minority_pct, 5),
                          "majority_class": str(majority), "majority_count": n_maj})]

    if ratio < 3:
        return [_ok("classification.imbalance", CLASSIF, "Classes are reasonably balanced",
                    f"Imbalance ratio {ratio:.1f}:1 (minority `{minority}` at {minority_pct:.1%}).",
                    "Balanced classes make accuracy a defensible headline metric.")]

    if n_min < 50:
        sev, extra = Severity.CRITICAL, (
            f" Only {n_min} minority-class rows exist, which is too few to estimate "
            f"minority performance reliably at all.")
    elif ratio > 20:
        sev, extra = Severity.CRITICAL, ""
    else:
        sev, extra = Severity.WARNING, ""

    return [Finding(
        check_id="classification.imbalance", category=CLASSIF, title="Class imbalance",
        severity=sev,
        what=f"Imbalance ratio is {ratio:.0f}:1 — minority class `{minority}` is "
             f"{minority_pct:.2%} of rows ({n_min:,} of {len(ctx.y_train):,}).{extra}",
        why=f"A model predicting `{majority}` for every row would score "
            f"{1 - minority_pct:.1%} accuracy while being useless. Accuracy and ROC-AUC "
            f"both look flattering at this ratio; neither reflects minority-class "
            f"performance.",
        recommendation=f"Report precision, recall and PR-AUC on class `{minority}` "
                       f"instead of accuracy. Try class_weight='balanced' first — it is "
                       f"simpler and leak-free — before reaching for resampling. If you "
                       f"do resample (imbalanced-learn), fit it inside the CV fold only.",
        columns=ev, weight=1.2)]


# --------------------------------------------------------------------- #
def _ok(cid: str, cat: str, title: str, what: str, why: str) -> Finding:
    return Finding(check_id=cid, category=cat, title=title, severity=Severity.PASSED,
                   what=what, why=why, recommendation="No action needed.")


def _skip(cid: str, cat: str, title: str) -> Finding:
    return Finding(check_id=cid, category=cat, title=title, severity=Severity.SKIPPED,
                   what="Prerequisites for this check were not met.",
                   why="This check could not run, so no assurance is provided either way.",
                   recommendation="Supply test= (and time_col=/entity_col= where relevant).")
