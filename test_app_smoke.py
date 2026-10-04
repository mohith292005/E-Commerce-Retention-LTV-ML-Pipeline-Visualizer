"""
End-to-end smoke test for app.py using Streamlit's AppTest framework.

Verifies that:
1. The app renders without exceptions (idle state).
2. Clicking "Run Pipeline Simulation" executes the full simulation
   (may take a few seconds due to intentional time.sleep pacing).
3. Post-simulation, the visualization canvas (charts, metrics, banner) renders.
"""

import sys
from streamlit.testing.v1 import AppTest

# Windows consoles default to cp1252, which cannot encode log glyphs (→, …)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    print("=" * 60)
    print("APP.SMOKE TEST")
    print("=" * 60)

    # ---- Phase 1: idle render ----
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    if at.exception:
        print(f"[FAIL] Idle render raised exception:\n{at.exception}")
        return 1
    print("[PASS] App renders in idle state without exceptions")
    print(f"       metrics rendered: {len(at.metric)}, plotly charts: {len(at.get('plotly_chart'))}")

    # ---- Phase 2: run simulation ----
    buttons = [b for b in at.button if "Run Pipeline" in (b.label or "")]
    if not buttons:
        print("[FAIL] 'Run Pipeline Simulation' button not found")
        return 1
    print("[PASS] Run button located; triggering simulation…")
    buttons[0].click()
    at.run(timeout=300)
    if at.exception:
        print(f"[FAIL] Simulation raised exception:\n{at.exception}")
        return 1
    print("[PASS] Simulation completed without exceptions")

    # ---- Phase 3: post-simulation state ----
    n_metrics = len(at.metric)
    n_charts = len(at.get("plotly_chart"))
    n_warnings = len(at.warning)
    n_errors = len(at.error)
    print(f"[INFO] metrics={n_metrics}, charts={n_charts}, "
          f"warnings={n_warnings}, errors={n_errors}")

    if n_errors > 0:
        print(f"[FAIL] Errors present after simulation: {[e.value for e in at.error]}")
        return 1

    # Expected: gauge + ltv + cluster charts (at least 3), metrics cards
    if n_charts < 3:
        print(f"[FAIL] Expected >=3 plotly charts after simulation, got {n_charts}")
        return 1
    if n_metrics < 4:
        print(f"[FAIL] Expected >=4 metric cards after simulation, got {n_metrics}")
        return 1

    # Stale-input warning should NOT appear right after a fresh run
    if n_warnings > 0:
        print(f"[WARN] Warnings present: {[w.value for w in at.warning]}")

    print("[PASS] Visualization canvas fully populated (charts + metric cards)")

    # ---- Phase 4: switch to manual sliders mode ----
    manual = None
    target_radio = None
    for radio in at.radio:
        for option in list(radio.options):
            if "Manual" in option:
                target_radio = radio
                manual = option
                break
        if target_radio is not None:
            break

    if target_radio is None:
        print(f"[FAIL] Manual mode option not found in any radio widget")
        return 1

    target_radio.set_value(manual)
    at.run(timeout=60)
    if at.exception:
        print(f"[FAIL] Manual mode raised exception:\n{at.exception}")
        return 1
    # Changing input after a run should raise the stale-results warning
    if len(at.warning) == 0:
        print("[WARN] Expected stale-input warning after changing mode")
    else:
        print("[PASS] Stale-input warning shown after mode switch")
    print("[PASS] Manual slider mode renders without exceptions")

    # ---- Phase 5: run simulation in manual mode ----
    buttons = [b for b in at.button if "Run Pipeline" in (b.label or "")]
    if not buttons:
        print("[FAIL] Run button not found in manual mode")
        return 1
    buttons[0].click()
    at.run(timeout=300)
    if at.exception:
        print(f"[FAIL] Manual-mode simulation raised exception:\n{at.exception}")
        return 1
    if len(at.error) > 0:
        print(f"[FAIL] Manual-mode errors: {[e.value for e in at.error]}")
        return 1
    if len(at.get("plotly_chart")) < 3:
        print("[FAIL] Manual-mode run did not render expected charts")
        return 1
    # The stale warning should be gone after a fresh run with current inputs
    stale = [w for w in at.warning if "Inputs changed" in w.value]
    if stale:
        print("[FAIL] Stale-input warning still present after fresh run")
        return 1
    print("[PASS] Manual slider simulation completed with full visualization")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
