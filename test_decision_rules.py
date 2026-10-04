"""Unit checks: business decision rules, median computation, single-record imputation."""

import pipeline as pl

cases = [(0.85, 0), (0.85, 2), (0.40, 0), (0.40, 1), (0.40, 2), (0.99, 1)]
expected = [
    "Trigger Automated 15% Discount Voucher",
    "Flag for Priority VIP Customer Support Outreach",
    "No Action Needed (Customer Active/Low Risk)",
    "No Action Needed (Customer Active/Low Risk)",
    "No Action Needed (Customer Active/Low Risk)",
    "No Action Needed (Customer Active/Low Risk)",
]

ok = True
for (churn, cid), exp in zip(cases, expected):
    d = pl.apply_decision_rules(churn, cid)
    status = "PASS" if d["action"] == exp else "FAIL"
    if d["action"] != exp:
        ok = False
    print(
        f"[{status}] churn={churn:.2f} cluster={cid} "
        f"({d['cluster_label']}) -> {d['severity']} | {d['action']}"
    )

med = pl.compute_feature_medians(pl.load_data())
print("[INFO] training medians:", {k: round(v, 1) for k, v in med.items()})

filled = pl.impute_missing_features(
    {"Age": float("nan"), "Total_Purchases": 5.0}, med
)
assert filled["Age"] == med["Age"], "imputation failed"
assert filled["Total_Purchases"] == 5.0, "non-NaN value must be preserved"
print(f"[PASS] impute_missing_features -> Age {filled['Age']:.0f} (median), "
      "Total_Purchases preserved")

# Error path: missing median must raise
try:
    pl.impute_missing_features({"NewCol": float("nan")}, med)
    print("[FAIL] expected ValueError for unknown column")
    ok = False
except ValueError as exc:
    print(f"[PASS] missing-median raises ValueError: {exc}")

print("UNIT CHECKS:", "PASSED" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
