# Getting Started with the preflightml Codebase

This doc is for reading the *code*, not just using the library. Read it in this order.

## 1. Install it and run the demo first

Before reading any code, see it work:

```bash
cd preflightml
pip install -e .          # "editable install" - lets you edit code and re-test instantly
python demo.py
```

You should see a boxed console report with a readiness score. Once you've seen the
output, the code below will make a lot more sense - you'll know what each piece is
building toward.

## 2. Read the files in this order

The codebase is small on purpose. Read these five files, in this order, and you'll
understand the whole architecture:

### Step 1 - `src/preflightml/core/finding.py`
Defines `Finding` - the shape every single check result takes (title, severity,
what/why/recommendation, affected columns). Every other file either creates one of
these or displays one. Nothing else to understand until you've read this.

### Step 2 - `src/preflightml/core/context.py`
Defines `InspectionContext` - takes your raw dataframe and figures out, once,
which columns are numeric vs categorical, which look like IDs, etc. Every check
receives one of these as input instead of the raw dataframe, so checks don't
each redo the same detective work.

### Step 3 - `src/preflightml/checks/basic.py`
Open this and find `check_imbalance` (search for `def check_imbalance`). This is
the simplest real check in the library - read it top to bottom. It:
1. Looks at `ctx.y_train` (the target column)
2. Counts how many rows are in each class
3. Decides CRITICAL / WARNING / PASSED based on the ratio
4. Builds and returns a `Finding`

Every other check in this file and in `leakage.py` follows this exact same pattern -
once `check_imbalance` makes sense, the rest is just "same idea, different math".

### Step 4 - `src/preflightml/core/registry.py`
Two jobs:
- `@register(...)` - the decorator above every check function. It's how a check
  "signs up" to be run automatically - nobody has to maintain a master list by hand.
- `compute_score(...)` - turns a list of Findings into the 0-100 readiness score.

### Step 5 - `src/preflightml/api.py`
The `inspect()` function you actually call as a user. Look at how short it is -
it just builds a `InspectionContext`, loops over every registered check, collects
the Findings, and wraps them in a `Report`. All the real logic lives in the files
above; this file just wires them together.

## 3. Everything else, once those five make sense

- `checks/leakage.py` - more checks, same pattern as Step 3
- `report.py` - the `Report` object: `.summary()`, `.explain(column)`, `.to_json()`, etc.
- `render/console.py` and `render/html.py` - turn a `Report` into text / HTML.
  These files never contain any detection logic - they only format `Finding` objects
  that already exist. That separation is deliberate (see "Why the folders are split
  this way" below).
- `tests/test_checks.py` - each test builds a small fake dataset with a known problem
  injected into it, runs `inspect()`, and checks the right Finding came back with the
  right severity. Reading the tests is often the fastest way to see what a check
  actually catches.

## 4. Why the folders are split the way they are

```
core/     <- the engine: what a Finding is, how scoring works. Knows nothing
             about leakage or imbalance specifically.
checks/   <- the actual detection logic. Each function: data in, Findings out.
render/   <- formatting only. Takes a finished Report, produces text/HTML/JSON.
```

This split means:
- Adding a new check later = one new function in `checks/`, nothing else changes.
- Adding a new output format (say, Slack messages) later = one new file in
  `render/`, nothing else changes.
- The scoring logic in `core/registry.py` never needs to know what a "leakage"
  check even is - it only ever deals with generic `Finding` objects.

## 5. A good first exercise

Try adding a brand new, very simple check yourself:

1. Open `checks/basic.py`
2. Copy `check_constant` as a template (it's short)
3. Write a check for something small, e.g. "flag any numeric column where the
   minimum value is negative but the column name suggests it shouldn't be
   (like `age` or `price`)"
4. Add `@register("your_check_id")` above it
5. Run `python demo.py` again - your check will run automatically without you
   touching `api.py` or `registry.py` at all

That's the payoff of the registry pattern: new checks just plug in.

## 6. If something breaks

- `pip install -e .` again if imports stop resolving after moving files
- `python -m pytest tests/ -q` to check you haven't broken an existing check
- Delete any stray `__pycache__` folders if Python seems to be running old code
  after an edit (`find . -name __pycache__ -exec rm -rf {} +`)
