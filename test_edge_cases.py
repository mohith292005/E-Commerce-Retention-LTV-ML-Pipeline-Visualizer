"""
Edge-case verification for app.py (AppTest).

1. Sample customer whose raw row contains NaN cells -> Stage 1 must impute
   with training medians instead of crashing on scaler.transform.
2. Manual slider mode -> Run button must execute a full simulation with the
   complete 5-feature vector (Age + Support Tickets included).
"""

import sys
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

# Windows consoles default to cp1252, which cannot encode log glyphs (→, …)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pipeline as pl


def find_nan_customer(picker_options) -> int:
    """Return a Customer_ID that is (a) offered by the picker and (b) missing
    at least one model feature in the raw CSV."""
    raw = pd.read_csv("data/synthetic_customers.csv")
    mask = raw[pl.MODEL_FEATURES].isna().any(axis=1)
    # AppTest may surface widget options as strings — normalise both sides
    try:
        option_ids = {int(x) for x in picker_options}
    except (TypeError, ValueError):
        option_ids = {str(x) for x in picker_options}
    ids = raw["Customer_ID"]
    ids = ids.astype(int) if not ids.isna().any() else ids
    candidates = raw.loc[mask & ids.isin(option_ids), "Customer_ID"]
    if candidates.empty:
        raise SystemExit("[SKIP] No NaN rows among picker options")
    return int(candidates.iloc[0])


def run_sim(at: AppTest) -> None:
    btn = [b for b in at.button if "Run Pipeline" in (b.label or "")][0]
    btn.click()
    at.run(timeout=300)


def main() -> int:
    # ---- Phase 1: NaN sample customer ----
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    if at.exception:
        print(f"[FAIL] idle: {at.exception}")
        return 1

    picker = next(sb for sb in at.selectbox if sb.label == "Customer ID")
    nan_id = find_nan_customer(picker.options)
    print(f"[INFO] Customer with NaN features (in test set): #{nan_id}")
    picker.set_value(str(nan_id))  # AppTest surfaces options as strings
    at.run(timeout=60)
    if at.exception:
        print(f"[FAIL] selecting NaN customer: {at.exception}")
        return 1

    run_sim(at)
    if at.exception:
        print(f"[FAIL] NaN-customer simulation: {at.exception}")
        return 1
    sim = at.session_state["sim_result"]
    print(f"[PASS] NaN-customer simulation OK -> segment={sim['cluster_label']}")
    stage1_lines = [l for l in sim["log_lines"] if "missing value" in l or "imputation" in l]
    print(f"       stage-1 log: {stage1_lines[:2]}")
    if not stage1_lines:
        print("[FAIL] Expected imputation log line for a NaN customer")
        return 1
    if not np.isfinite(sim["churn_probability"]):
        print("[FAIL] churn probability not finite")
        return 1
    print(f"       churn={sim['churn_probability']:.4f} ltv={sim['predicted_ltv']:.2f} "
          f"action={sim['decision']['action']}")

    # ---- Phase 2: manual slider mode simulation ----
    at2 = AppTest.from_file("app.py", default_timeout=120)
    at2.run()
    if at2.exception:
        print(f"[FAIL] idle(2): {at2.exception}")
        return 1

    radio = next(
        r for r in at2.radio
        if any("Manual" in str(o) for o in r.options)
    )
    radio.set_value(next(str(o) for o in radio.options if "Manual" in str(o)))
    at2.run(timeout=60)
    if at2.exception:
        print(f"[FAIL] switching to manual mode: {at2.exception}")
        return 1

    # Push sliders to an extreme at-risk profile
    settings = {
        "Age": 64,
        "Days Since Last Purchase": 350,
        "Total Purchases": 1,
        "Avg Order Value ($)": 25.0,
        "Support Tickets": 9,
    }
    for sl in at2.slider:
        for label, value in settings.items():
            if sl.label == label:
                sl.set_value(value)
    at2.run(timeout=60)
    if at2.exception:
        print(f"[FAIL] setting sliders: {at2.exception}")
        return 1

    run_sim(at2)
    if at2.exception:
        print(f"[FAIL] manual-mode simulation: {at2.exception}")
        return 1
    sim2 = at2.session_state["sim_result"]
    feats = sim2["target_features"]
    print(f"[PASS] Manual-mode simulation OK -> features={ {k: round(v,1) for k, v in feats.items()} }")
    if set(feats) != set(pl.MODEL_FEATURES):
        print(f"[FAIL] target_features keys mismatch: {sorted(feats)}")
        return 1
    print(f"       churn={sim2['churn_probability']:.4f} segment={sim2['cluster_label']} "
          f"action={sim2['decision']['action']}")
    print(f"       first log line: {sim2['log_lines'][0]}")
    allowed = {
        "Trigger Automated 15% Discount Voucher",
        "Flag for Priority VIP Customer Support Outreach",
        "No Action Needed (Customer Active/Low Risk)",
    }
    if sim2["decision"]["action"] not in allowed:
        print(f"[FAIL] Unexpected action: {sim2['decision']['action']}")
        return 1

    print("=" * 60)
    print("EDGE CASES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
