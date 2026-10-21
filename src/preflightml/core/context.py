"""Shared context passed to every check."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

# Heuristic name fragments used when auto-detecting column roles.
_ENTITY_HINTS = ("customer", "user", "client", "account", "member", "patient", "entity","household")
_ID_HINTS = ("_id", "id_", "uuid", "guid", "hash", "ref", "key")
_TIME_HINTS = ("date", "time", "timestamp", "_at", "_dt", "created", "updated")


@dataclass
class InspectionContext:
    """Everything a check needs, computed once."""

    train: pd.DataFrame
    target: str
    test: pd.DataFrame | None = None
    time_col: str | None = None
    entity_col: str | None = None

    # Populated in __post_init__
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    datetime_cols: list[str] = field(default_factory=list)
    id_like_cols: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target not in self.train.columns:
            raise ValueError(
                f"target column {self.target!r} not found in the training frame. "
                f"Available columns: {list(self.train.columns)[:10]}..."
            )
        self._classify_columns()
        self._infer_roles()

    # ------------------------------------------------------------------ #
    @property
    def features(self) -> list[str]:
        return [c for c in self.train.columns if c != self.target]

    @property
    def has_test(self) -> bool:
        return self.test is not None

    @property
    def y_train(self) -> pd.Series:
        return self.train[self.target]

    def _classify_columns(self) -> None:
        for col in self.features:
            s = self.train[col]
            if ptypes.is_datetime64_any_dtype(s):
                self.datetime_cols.append(col)
            elif ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s):
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)

            # id-like: near-unique, or name looks like an identifier.
            # Continuous floats are naturally near-unique and are NOT identifiers,
            # so they only qualify if the column name also looks like one.
            n = len(s)
            if n > 0:
                uniq_ratio = s.nunique(dropna=True) / n
                name_hit = any(h in col.lower() for h in _ID_HINTS)
                is_continuous = ptypes.is_float_dtype(s)
                if (uniq_ratio > 0.95 and not is_continuous) or (name_hit and uniq_ratio > 0.5):
                    self.id_like_cols.append(col)

    def _infer_roles(self) -> None:
        """Auto-detect time_col / entity_col, recording the assumption made.

        Detection is best-effort. Every inference is recorded so the report can
        state it explicitly and downgrade confidence on dependent findings.
        """
        if self.time_col is None:
            candidates = list(self.datetime_cols) or [
                c for c in self.features if any(h in c.lower() for h in _TIME_HINTS)
            ]
            if len(candidates) == 1:
                self.time_col = candidates[0]
                self.assumptions.append(
                    f"time_col was not supplied; inferred as {self.time_col!r}. "
                    f"Pass time_col= explicitly to confirm or override."
                )
            elif len(candidates) > 1:
                self.assumptions.append(
                    f"Multiple possible time columns found ({candidates}); temporal "
                    f"checks were skipped. Pass time_col= to enable them."
                )

        if self.entity_col is None:
            candidates = [
                c for c in self.features
                if any(h in c.lower() for h in _ENTITY_HINTS)
                and self.train[c].nunique(dropna=True) < len(self.train)
            ]
            if len(candidates) == 1:
                self.entity_col = candidates[0]
                self.assumptions.append(
                    f"entity_col was not supplied; inferred as {self.entity_col!r}. "
                    f"Pass entity_col= explicitly to confirm or override."
                )
            elif len(candidates) > 1:
                self.assumptions.append(
                    f"Multiple possible entity columns found ({candidates}); entity "
                    f"leakage check was skipped. Pass entity_col= to enable it."
                )

    def inferred(self, role: str) -> bool:
        """True if the given role was guessed rather than supplied."""
        return any(role in a for a in self.assumptions)
