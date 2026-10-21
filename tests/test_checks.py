import numpy as np
import pandas as pd
import pytest

from preflightml import inspect, Severity


def _clean(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(5, 2, n),
                       "c": rng.choice(list("xyz"), n)})
    df["t"] = (rng.random(n) < 0.45).astype(int)
    return df


def _sev(report, check_id):
    return next(f.severity for f in report.findings if f.check_id == check_id)


def test_clean_dataset_scores_high():
    df = _clean()
    r = inspect(df, target="t")
    assert r.score >= 90
    assert not r.has_blocking_issues


def test_missing_target_raises():
    with pytest.raises(ValueError):
        inspect(_clean(), target="nope")


def test_split_checks_skip_not_pass_without_test():
    r = inspect(_clean(), target="t")
    assert _sev(r, "leakage.row_overlap") is Severity.SKIPPED
    assert _sev(r, "compat.numeric_shift") is Severity.SKIPPED


def test_target_leakage_detected():
    df = _clean()
    df["leaky"] = df["t"] * 3 + 1
    r = inspect(df, target="t")
    f = next(x for x in r.findings if x.check_id == "leakage.target_association")
    assert f.severity is Severity.CRITICAL
    assert "leaky" in f.affected_columns


def test_imbalance_detected_and_names_minority():
    rng = np.random.default_rng(1)
    df = _clean()
    df["t"] = (rng.random(len(df)) < 0.01).astype(int)
    r = inspect(df, target="t")
    f = next(x for x in r.findings if x.check_id == "classification.imbalance")
    assert f.severity is Severity.CRITICAL
    assert f.columns[0].detail["minority_class"] == "1"


def test_entity_overlap_detected():
    df = _clean()
    df["customer_id"] = np.arange(len(df)) % 200
    tr = df.sample(2000, random_state=1)
    te = df.drop(tr.index)
    r = inspect(tr, target="t", test=te, entity_col="customer_id")
    assert _sev(r, "leakage.entity_overlap") is Severity.CRITICAL


def test_temporal_leakage_detected():
    df = _clean()
    df["ts"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(len(df)), "h")
    tr = df.sample(2000, random_state=1)
    te = df.drop(tr.index)
    r = inspect(tr, target="t", test=te, time_col="ts")
    assert _sev(r, "leakage.temporal") is Severity.CRITICAL


def test_row_overlap_runs_with_missing_values():
    """Regression: on pandas 3.x this check used to crash on NaN and was
    silently reported as Skipped."""
    df = _clean()
    df.loc[::7, "a"] = np.nan
    df.loc[::5, "c"] = None
    tr = df.sample(2000, random_state=1)
    te = df.drop(tr.index).iloc[:500]
    r = inspect(tr, target="t", test=te)
    assert _sev(r, "leakage.row_overlap") is Severity.PASSED


def test_row_overlap_detected_with_missing_values():
    df = _clean()
    df.loc[::7, "a"] = np.nan
    tr = df.sample(2000, random_state=1)
    te = pd.concat([df.drop(tr.index).iloc[:500], tr.iloc[:100]])
    r = inspect(tr, target="t", test=te)
    assert _sev(r, "leakage.row_overlap") is Severity.CRITICAL


def test_no_check_is_skipped_by_an_internal_error():
    """A check that crashes is reported as Skipped. With a full set of inputs,
    nothing should be skipped for that reason."""
    df = _clean()
    df["customer_id"] = np.arange(len(df)) % 300
    df["ts"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(len(df)), "h")
    tr = df.sample(2000, random_state=1)
    te = df.drop(tr.index)
    r = inspect(tr, target="t", test=te, time_col="ts", entity_col="customer_id")
    crashed = [f.check_id for f in r.skipped if "Error" in f.what]
    assert crashed == []


def test_score_breakdown_reconstructs_score():
    df = _clean()
    df["leaky"] = df["t"]
    r = inspect(df, target="t")
    total = sum(b["points_lost"] for b in r.score_breakdown())
    assert abs((100 - total) - r.score) < 1.0


def test_column_report_inverts_findings():
    df = _clean()
    df["leaky"] = df["t"]
    r = inspect(df, target="t")
    assert "leaky" in r.column_report()


def test_outputs_render():
    r = inspect(_clean(), target="t")
    assert "PREFLIGHTML REPORT" in __import__("preflightml").render.console.render_summary(r)
    assert r.to_json().startswith("{")
    assert r.to_html().startswith("<!DOCTYPE html>")