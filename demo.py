"""Demo: build a deliberately broken dataset and inspect it."""
import numpy as np, pandas as pd
from preflightml import inspect

rng = np.random.default_rng(7)
N = 6000
base = pd.DataFrame({
    "customer_id": rng.integers(0, 2000, N),
    "signup_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 400, N), "D"),
    "income": rng.normal(60000, 15000, N),
    "tenure": rng.integers(0, 120, N),
    "postcode": rng.integers(0, 900, N).astype(str),
    "plan": rng.choice(["basic", "pro", "enterprise"], N, p=[.6, .3, .1]),
    "legacy_flag": ["Y"] * N,
    "orders_30d": rng.poisson(3, N),
})
base["orders_30d_copy"] = base["orders_30d"] * 1.0 + rng.normal(0, .01, N)
base["converted"] = (rng.random(N) < 0.04).astype(int)
base["cancellation_reason"] = np.where(base["converted"] == 1, "n/a", "churned_" + base["plan"])

train = base.sample(4500, random_state=1)
test = base.drop(train.index)
train = pd.concat([train, test.sample(150, random_state=2)])   # inject row overlap
test["income"] = test["income"] * 1.6                          # inject drift

report = inspect(train, target="converted", test=test,
                 time_col="signup_date", entity_col="customer_id")
report.summary()
report.details(severity="critical")
print("\n--- drill-down ---")
report.explain("income")
print("\n--- CI gate ---")
print("blocking:", report.has_blocking_issues)
print("\n--- score breakdown ---")
for b in report.score_breakdown():
    print(" ", b)
report.to_html("report.html")
