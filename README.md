# preflightml

**Run this before you train your classifier.**

`preflightml` answers one question: *is my classification dataset ready to model, and if not, exactly what should I fix?*

It inspects your data before any model is trained and reports, column by column, what is risky, why it matters and what to do about it. It is not an AutoML tool, a modelling framework or a model trainer. It runs *before* those, and tells you whether the numbers they produce can be trusted.

- **One call, no setup.** `inspect(df, target="churn")` - no schema objects, no configuration.
- **Column-level answers.** Every finding names the exact columns responsible and the statistic that triggered it.
- **A decision, not just statistics.** A 0-100 readiness score, a Critical / Warning / Passed breakdown, and a KEEP / INVESTIGATE / REMOVE list of features.
- **Honest about what it did not check.** A check that could not run is reported as *Skipped*, never as *Passed*.
- **Lightweight.** Depends only on `pandas` and `numpy`.

## Installation

```bash
pip install preflightml
```

Requires Python 3.10 or newer and pandas 2.0 or newer. Tested on Python 3.12 and 3.13 with pandas 2.3 and 3.0.

## Quickstart

### 1. Inspect a whole dataset

```python
import pandas as pd
import preflightml as pf

df = pd.read_csv("churn.csv")

report = pf.inspect(df, target="Churn")
report.summary()
```

> Tip: `import preflightml as pf` avoids shadowing Python's built-in `inspect` module. `from preflightml import inspect` also works.

Real output on a synthetic telecom churn dataset (8,000 rows, with one planted leaky column):

```
+============================================================+
|                     PREFLIGHTML REPORT                     |
+============================================================+
| Dataset             8,000 x 16                             |
| MODEL READINESS     55 / 100                               |
|                                                            |
| Critical            1                                      |
| Warnings            3                                      |
| Passed              7                                      |
| Skipped             6  (not assessed)                      |
+------------------------------------------------------------+
| TARGET                                                     |
|                                                            |
| Target              Churn                                  |
| Classes             2                                      |
|   class 0           69.9%                                  |
|   class 1           30.1%                                  |
+------------------------------------------------------------+
| FEATURE RISK                                               |
|                                                            |
| [!] Potential target leakage                               |
|    -> CancellationTicketID                                 |
| [~] High-cardinality categorical features                  |
|    -> CancellationTicketID                                 |
| [~] Identifier-like columns used as features               |
|    -> customerID                                           |
+------------------------------------------------------------+
| TARGET PATTERNS                                            |
|                                                            |
| - class 1 averages 21% lower `tenure` than class 0         |
| - class 1 averages 10% higher `MonthlyCharges` than        |
| class 0                                                    |
+------------------------------------------------------------+
| RECOMMENDED FEATURES                                       |
|                                                            |
| KEEP                13 features                            |
| INVESTIGATE         customerID                             |
| REMOVE              CancellationTicketID                   |
+------------------------------------------------------------+
| RECOMMENDATION                                             |
|                                                            |
| DO NOT TRAIN FINAL MODEL YET                               |
+============================================================+
```

(Shortened here; the real report also includes data-quality details, recommendations per finding and any assumptions made.)

### 2. Check the quality of your train/test split

Pass your held-out set to also check split integrity and train/test compatibility:

```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2, random_state=1, stratify=df["Churn"])

report = pf.inspect(
    train,
    target="Churn",
    test=test,
    time_col="signup_date",     # optional: enables temporal leakage checks
    entity_col="customerID",    # optional: enables entity-overlap checks
)
report.summary()
```

Without `test=`, the six split-dependent checks are reported as **Skipped**, so a dataset is never shown as "clean" for checks that did not run.

### 3. Drill down

```python
report.details("critical")             # full what / why / action for every Critical finding
report.explain("CancellationTicketID")  # everything wrong with one column, with statistics
report.recommended_features()           # {"keep": [...], "investigate": [...], "remove": [...]}
report.column_report()                  # column -> list of issues, as a dict
report.score_breakdown()                # every point deducted from the score
```

### 4. Export

```python
report.to_html("preflight_report.html")   # standalone file, no external assets
report.to_json("preflight_report.json")   # machine-readable
```

### 5. Use it as a CI/CD gate

```python
report = pf.inspect(train, target="Churn", test=test)
report.to_json("preflight_report.json")

if report.has_blocking_issues:      # True if any Critical finding exists
    raise SystemExit("Preflight failed - see preflight_report.json")
```

## API

```python
preflightml.inspect(
    train,               # DataFrame, including the target column
    target,              # name of the target column
    test=None,           # optional held-out DataFrame (enables split checks)
    time_col=None,       # optional event-time column (enables temporal leakage check)
    entity_col=None,     # optional entity column, e.g. customer_id (enables entity-overlap check)
) -> Report
```

If `time_col` or `entity_col` is not given, the library tries to infer it when exactly one obvious candidate exists. Any such guess is listed under **ASSUMPTIONS MADE** in the report, and findings that depend on it carry reduced confidence. Pass them explicitly when you know them.

**`Report` attributes and methods**

| Member | What it gives you |
|---|---|
| `score` | Model readiness score, 0-100 |
| `verdict` | Plain-language decision, e.g. `DO NOT TRAIN FINAL MODEL YET` |
| `critical`, `warnings`, `passed`, `skipped` | Lists of findings by outcome |
| `has_blocking_issues` | `True` if any Critical finding exists |
| `summary()` | Console summary |
| `details(severity=None)` | Full findings, optionally for one severity |
| `explain(column)` | Drill-down for one column |
| `recommended_features()` | KEEP / INVESTIGATE / REMOVE lists |
| `target_patterns()` | Plain-language relationships between top features and the target |
| `column_report()` | Column -> issues mapping |
| `score_breakdown()` | Itemised deductions behind the score |
| `to_html(path)`, `to_json(path)`, `to_dict()` | Exports |

Every finding has the same shape: `check_id`, `category`, `title`, `severity`, `what`, `why`, `recommendation`, `confidence`, and per-column `evidence`.

## What it checks

| Category | Check | Flagged when |
|---|---|---|
| Dataset & Target Health | Missing values | A feature is >40% missing (Critical above 80%) |
| | Duplicate rows | Any fully duplicated rows (Critical above 5%) |
| | Constant columns | One value in >99% of rows |
| | Target quality | Target has missing values or fewer than 2 classes (Critical) |
| | Sample size | Fewer than 50 rows per feature (Critical below 10) |
| Leakage & Split Integrity | Target leakage | Feature has rank correlation >= 0.95 with the target (Critical); 0.80-0.95 is a Warning |
| | Identifier columns | Near-unique or identifier-named columns used as features |
| | Train/test row overlap | Identical rows in both sets (Critical above 1%) |
| | Entity overlap | Same entity in train and test (Critical above 5%) |
| | Temporal leakage | Training rows dated after the earliest test row (Critical above 5%) |
| Train/Test Compatibility | Numeric shift | Population Stability Index > 0.1 (Critical above 0.25) |
| | Unseen categories | Categories in test that never appear in train |
| | Target shift | Class proportions differ by 5 points or more |
| Feature Risk | High cardinality | A categorical feature has more than 50 distinct values |
| | Redundancy | Numeric feature pairs with absolute correlation > 0.95 |
| Classification Risk | Class imbalance | Majority:minority ratio of 3:1 or more (Critical above 20:1 or fewer than 50 minority rows) |
| Target Patterns | Directional summaries | Informational only; never affects the score |

Split-dependent checks (row overlap, entity overlap, temporal leakage, numeric shift, unseen categories, target shift) need `test=`.

## How the score works

The score is deterministic and can be reconstructed by hand:

- It starts at 100.
- Each Critical finding deducts `18 x weight x confidence` points; each Warning deducts `6 x weight x confidence`.
- Any Critical finding caps the score at 60; warnings alone cap it at 90.
- `report.score_breakdown()` lists every deduction.

Verdicts: any Critical -> `DO NOT TRAIN FINAL MODEL YET`; score 90+ -> `READY TO TRAIN`; otherwise `TRAIN WITH CAUTION - REVIEW WARNINGS`.

## Known limitations (v0.1)

This is an early release. Please read these before relying on it:

- **Leakage detection is statistical.** It catches the *symptom* of leakage (a feature that is suspiciously predictive). It cannot know whether a feature will actually be available at prediction time. A leak that is not statistically obvious will not be found.
- **Categorical association is approximate.** Non-numeric columns are encoded as category codes before a rank correlation is computed, so for nominal categoricals with many levels the strength can be under- or over-estimated.
- **A random split will trigger the temporal check.** If you split with `train_test_split`, some training rows will naturally be dated after some test rows. That is expected; the flag is only a real problem if you intended a chronological split.
- **Thresholds are fixed defaults** and not yet configurable. They have not been validated across a wide range of real datasets, so expect some false positives and false negatives.
- **Missing-value advice is generic.** It does not yet check whether a column's missingness is itself associated with the target (in which case adding an "is missing" indicator would leak the target).
- **Target patterns support binary targets only.** Other checks are designed for classification in general, but multi-class targets are less tested.
- Tested on Python 3.12 and 3.13 with pandas 2.3 and 3.0 only. Other versions are untested.

## Contributing

Issues and pull requests are welcome at https://github.com/AvinAshok2050/preflightml.git.

```bash
git clone https://github.com/AvinAshok2050/preflightml.git
cd preflightml
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Each check is a small function that takes the dataset context and returns findings; adding one means adding one function and registering it.

## License

MIT. See [LICENSE](LICENSE).

Copyright (c) 2026 Avin Ashok