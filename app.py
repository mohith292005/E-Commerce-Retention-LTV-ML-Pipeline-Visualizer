"""
app.py
======
ML Pipeline & Customer Analytics Command Center
================================================
Modern, card-based Streamlit dashboard for the E-Commerce
Customer Retention & LTV ML Pipeline — RetainIQ visual design.

Architecture
------------
* No native sidebar — all controls live inside a floating popover
* Sticky top header with segmented_control for global state
* Triple-column card grid (Target Profile / Pipeline Feed / Terminal)
* st.status() reveals pipeline execution layers in real time
* Expansive Plotly behavioural-segmentation footer

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pipeline as pl
from data_generator import generate_synthetic_customer_data, save_data_to_csv

# =====================================================================
# 1. PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="RetainIQ — Customer Intelligence & Pipeline Engine",
    page_icon="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Crect x='2' y='3' width='9' height='8' rx='2'/%3E%3Crect x='13' y='3' width='9' height='8' rx='2'/%3E%3Crect x='2' y='13' width='9' height='8' rx='2' opacity='.55'/%3E%3Crect x='13' y='13' width='9' height='8' rx='2' opacity='.55'/%3E%3Ccircle cx='12' cy='12' r='1.6'/%3E%3C/svg%3E",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# 2. RETAINIQ DESIGN SYSTEM — CSS injection (style-only, no layout gap)
# =====================================================================
RETAINIQ_CSS = """
<style>
/* 1. Force pure black — never flash white */
html, body, #root, .stApp, [data-testid="stAppViewContainer"] {
  background: #000000 !important; color: #ffffff !important;
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
  overflow-x: hidden;
}
/* 2. Font stacks */
body, .stMarkdown, h1, h2, h3, p, label, .stMetric > label {
  font-family: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif !important;
}
.em-italic {
  font-family: "Instrument Serif", "Times New Roman", Times, serif !important;
  font-style: italic; letter-spacing: -0.03em;
}
/* 3. Neutralise vertical-block gaps */
[data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.block-container { background: transparent !important; padding-top: 0 !important; padding-bottom: 0 !important; }
/* 4. Card containers */
[data-testid="stVerticalBlockBorderVisible"] {
  background: rgba(15,15,15,0.55) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 14px !important;
}
/* 5. Liquid-glass buttons */
.stButton > button {
  border-radius: 6px !important;
  border: 1px solid rgba(255,255,255,0.55) !important;
  height: 42px !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
  letter-spacing: -0.02em !important;
  white-space: nowrap !important;
  background: linear-gradient(180deg, #ffffff 0%, #e7e7e7 48%, #cfcfcf 100%) !important;
  color: #111 !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.95) !important;
  transition: all 0.35s ease !important;
  width: 100% !important;
}
.stButton > button:hover {
  background: linear-gradient(180deg, #fff 0%, #f3f6ff 42%, #d5def2 100%) !important;
  border-color: #f2f6ff !important;
  box-shadow: inset 0 1px 0 #fff, 0 0 22px rgba(186,208,255,0.35), 0 8px 18px rgba(255,255,255,0.12) !important;
}
/* 6. Hide default Streamlit chrome */
header[data-testid="stHeader"], #MainMenu, stFooter, [data-testid="stToolbar"] {
  display: none !important;
}
/* 7. Grain overlay */
body::before {
  content: ""; position: fixed; inset: 0; z-index: 9999; pointer-events: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' opacity='0.045'/%3E%3C/svg%3E");
  background-size: 200px 200px;
}
/* 8. Metrics */
[data-testid="stMetric"] {
  background: rgba(15,15,15,0.5) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 10px !important;
  padding: 10px 14px !important;
}
[data-testid="stMetricLabel"] { color: #9a9a9a !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700 !important; font-size: 1.3rem !important; }
[data-testid="stMetricDelta"] { color: #9a9a9a !important; }
/* 9. Terminal log */
.terminal-log {
  background: rgba(6,6,6,0.9) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 10px !important;
  font-family: "SF Mono", "Fira Code", "JetBrains Mono", monospace !important;
  font-size: 0.78rem !important;
  line-height: 1.65 !important;
  color: #d8d8d8 !important;
  max-height: 420px;
  overflow-y: auto;
  padding: 14px 16px !important;
}
.terminal-log pre { margin: 0 !important; white-space: pre-wrap !important; word-wrap: break-word !important; }
/* 10. Dataframe */
[data-testid="stDataFrame"] { background: rgba(15,15,15,0.8) !important; }
/* 11. Segmented control */
[data-testid="stSegmentedControl"] { background: transparent !important; }
/* 12. Status container */
[data-testid="stStatusWidget"] {
  background: rgba(15,15,15,0.5) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 12px !important;
}
/* 13. Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.22); }
/* 14. Popover */
section[data-testid="stPopover"] { background: rgba(8,8,8,0.95) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 12px !important; }
/* 15. Plotly fix */
.js-plotly-plot .plotly .modebar { display: none !important; }
/* 16. Equal card heights — force flex-stretch on 3-col cards */
[data-testid="stContainer"] { display: flex !important; flex-direction: column !important; height: 100% !important; }
[data-testid="stContainer"] > div { flex-grow: 1 !important; }
/* 17. Profile metric grid uniform height */
.profile-metric-pair { display: flex; gap: 8px; align-items: stretch; }
.profile-metric-pair > div { flex: 1 1 0; min-height: 58px; }
/* 18. Terminal fixed-height scroll window */
.terminal-scroll { max-height: 380px; overflow-y: auto; }
/* 19. Controls horizontal baseline alignment */
.stButton, [data-testid="stSegmentedControl"], [data-testid="stSelectbox"] { vertical-align: middle !important; }
.stButton > button { margin-top: 2px !important; }
/* 20. Footer plot toggle spacing */
[data-testid="stWidgetLabel"] + div { margin-bottom: 10px !important; }
/* 21. Matching column/container heights */
[data-testid="stHorizontalBlock"] { display: flex !important; align-items: stretch !important; }
[data-testid="stHorizontalBlock"] > div { flex: 1 1 auto !important; display: flex !important; flex-direction: column !important; min-height: 0 !important; }
[data-testid="stHorizontalBlock"] > div > div { flex-grow: 1 !important; }
/* 22. Uniform metric height/padding in 5-col row */
[data-testid="stMetric"] { width: 100%; min-height: 88px; }
/* 23. Table cell padding */
div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th,
div[data-testid="stTable"] td, div[data-testid="stTable"] th {
  padding-top: 8px !important;
  padding-bottom: 8px !important;
}
</style>
"""
st.html(RETAINIQ_CSS)

# =====================================================================
# 3. CONSTANTS
# =====================================================================
FEATURE_COLS: List[str] = list(pl.MODEL_FEATURES)
PLOT_FEATURES: List[str] = list(pl.CONTINUOUS_FEATURES)
CLUSTER_LABELS: Dict[int, str] = pl.CLUSTER_LABELS
CHURN_THRESHOLD: float = pl.CHURN_THRESHOLD

def _cluster_col_indices(bundle: Dict[str, Any]) -> List[int]:
    feature_cols = list(bundle.get("feature_cols", FEATURE_COLS))
    cluster_cols = list(bundle.get("cluster_feature_cols", PLOT_FEATURES))
    return [feature_cols.index(c) for c in cluster_cols]

SEGMENT_COLORS: Dict[str, str] = {
    CLUSTER_LABELS[0]: "#e74c3c",
    CLUSTER_LABELS[1]: "#f39c12",
    CLUSTER_LABELS[2]: "#2980b9",
}

SIM_SPEEDS: Dict[str, float] = {
    "Fast ⚡": 0.10,
    "Normal 🚀": 0.30,
    "Detailed 🐢": 0.65,
}

# =====================================================================
# 4. CACHED DATA & MODEL LOADING
# =====================================================================
@st.cache_data(show_spinner=False)
def load_or_generate_data() -> pd.DataFrame:
    try:
        return pl.load_data("data/synthetic_customers.csv")
    except FileNotFoundError:
        df = generate_synthetic_customer_data(n_customers=1000, random_seed=42)
        save_data_to_csv(df, filepath="data/synthetic_customers.csv")
        return df

@st.cache_resource(show_spinner=False)
def train_pipeline_bundle() -> Dict[str, Any]:
    load_or_generate_data()
    return pl.run_full_pipeline()

# =====================================================================
# 5. FORMAT HELPERS
# =====================================================================
def _elapsed(start_ts: float) -> str:
    delta = max(0.0, time.time() - start_ts)
    minutes, seconds = divmod(delta, 60.0)
    return f"[{int(minutes):02d}:{seconds:05.2f}]"

def _fmt_money(value: float) -> str:
    if pd.isna(value):
        return "NaN → median"
    return f"${float(value):,.2f}"

def _fmt_feature(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return "⚡ imputed"
    return f"{float(value):g}{suffix}"

# =====================================================================
# 6. ML FUNCTIONS — preserved verbatim from original
# =====================================================================
def render_decision_banner(decision: Dict[str, Any]) -> None:
    action = decision["action"]
    severity = decision["severity"]
    churn_pct = decision["churn_probability"] * 100
    if severity == "high":
        if decision["cluster_id"] == 0:
            icon, gradient = "🎟️", "linear-gradient(90deg, #e74c3c 0%, #e67e22 100%)"
            rule = "IF Churn > 70% AND Cluster == At-Risk Bargain Hunters"
        else:
            icon, gradient = "👑", "linear-gradient(90deg, #8e44ad 0%, #e74c3c 100%)"
            rule = "IF Churn > 70% AND Cluster == VIP High Spenders"
    else:
        icon, gradient = "✅", "linear-gradient(90deg, #27ae60 0%, #2ecc71 100%)"
        rule = "ELSE branch (churn <= 70% or neutral segment)"
    st.markdown(
        f"""
        <div style="
            background: {gradient};
            border-radius: 16px;
            padding: 22px 28px;
             color: #ffffff;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            margin-top: 6px;">
            <div style="font-size: 0.85rem; opacity: 0.9; letter-spacing: 1.5px;">
                STEP 3 · BUSINESS LOGIC &amp; INTERVENTION ENGINE
            </div>
            <div style="font-size: 1.55rem; font-weight: 700; margin-top: 8px;">
                {icon} {action}
            </div>
            <div style="font-size: 0.92rem; margin-top: 10px; opacity: 0.95;">
                Rule fired: <b>{rule}</b> &nbsp;·&nbsp;
                Churn probability: <b>{churn_pct:.1f}%</b> &nbsp;·&nbsp;
                Segment: <b>{decision["cluster_label"]}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def build_churn_gauge(churn_probability: float, baseline_pct: float) -> go.Figure:
    pct = float(np.clip(churn_probability * 100.0, 0.0, 100.0))
    bar_color = "#10b981" if pct <= 50 else ("#f59e0b" if pct <= 70 else "#ef4444")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            delta={"reference": baseline_pct, "suffix": " pp vs baseline"},
            number={"suffix": "%", "font": {"size": 46}},
            title={"text": "Churn Probability", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color, "thickness": 0.35},
                "bgcolor": "rgba(255,255,255,0)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(16,185,129,0.35)"},
                    {"range": [50, 70], "color": "rgba(245,158,11,0.35)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.40)"},
                ],
                "threshold": {"line": {"color": "#dc2626", "width": 4}, "thickness": 0.75, "value": pct},
            },
        )
    )
    fig.update_layout(height=340, margin=dict(t=70, b=10, l=30, r=30), font=dict(color="#e5e7eb"),
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig

def build_ltv_comparison_chart(predicted_ltv: float, historical_avg: float) -> go.Figure:
    fig = go.Figure(data=[go.Bar(
        name="Value", x=["Historical Avg", "Predicted LTV"],
        y=[historical_avg, predicted_ltv],
        marker_color=["#64748b", "#3b82f6"],
        text=[f"${historical_avg:,.0f}", f"${predicted_ltv:,.0f}"],
        textposition="outside",
    )])
    fig.update_layout(height=300, margin=dict(t=30, b=10, l=10, r=10),
                      yaxis_title="Total LTV ($)", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e7eb"),
                      showlegend=False, bargap=0.45)
    return fig

def _attach_pulse_animation(fig: go.Figure, highlight_idx: int, trace_builder, base_size: int = 20) -> None:
    base_traces = [t.to_plotly_json() for t in fig.data]
    sizes = [base_size, base_size + 16, base_size, base_size + 8]
    frames = []
    for i, size in enumerate(sizes):
        frame_data: List[Any] = [dict(bt) for bt in base_traces]
        frame_data[highlight_idx] = trace_builder(size).to_plotly_json()
        frames.append(go.Frame(data=frame_data, name=f"pulse-{i}"))
    fig.frames = tuple(frames)
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="left", x=0.0, xanchor="left", y=1.08, yanchor="top",
        bgcolor="rgba(59,130,246,0.85)", font=dict(color="#ffffff", size=12),
        buttons=[
            dict(label="▶ Pulse Target", method="animate",
                 args=[None, dict(frame=dict(duration=450, redraw=True), fromcurrent=True, transition=dict(duration=350))]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
        ],
    )])

def build_cluster_figure(bundle: Dict[str, Any], target_cluster_scaled: np.ndarray,
                         cluster_id: int, target_label: str, view: str) -> go.Figure:
    data = bundle["data"]
    labels = bundle["predictions"]["cluster_labels"]
    X_scaled = bundle["scaler"].transform(data[FEATURE_COLS])
    X_cluster = X_scaled[:, _cluster_col_indices(bundle)]
    viz_df = pd.DataFrame({
        "Days_Since_Last_Purchase": X_cluster[:, 0],
        "Total_Purchases": X_cluster[:, 1],
        "Avg_Order_Value": X_cluster[:, 2],
        "Segment": [CLUSTER_LABELS.get(int(c), "Unknown") for c in labels],
    })
    tx, ty, tz = (float(v) for v in target_cluster_scaled.flatten())
    target_segment = CLUSTER_LABELS.get(cluster_id, "Unknown")
    highlight_color = SEGMENT_COLORS.get(target_segment, "#ffffff")

    if view == "3D":
        fig = px.scatter_3d(viz_df, x="Days_Since_Last_Purchase", y="Total_Purchases", z="Avg_Order_Value",
                            color="Segment", color_discrete_map=SEGMENT_COLORS, opacity=0.55,
                            title="K-Means Customer Segments (Standardized Feature Space)")
        fig.add_trace(go.Scatter3d(x=[tx], y=[ty], z=[tz], mode="markers",
                                   marker=dict(size=26, color=highlight_color, opacity=0.30),
                                   name="Target Halo", hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=[tx], y=[ty], z=[tz], mode="markers",
                                   marker=dict(size=20, color=highlight_color, symbol="diamond", line=dict(width=3, color="#ffffff")),
                                   name=f"Target: {target_label}",
                                   hovertemplate=f"<b>{target_label}</b><br>Days since last purchase: %{{x:.2f}}σ<br>Total purchases: %{{y:.2f}}σ<br>Avg order value: %{{z:.2f}}σ<extra></extra>"))
        highlight_idx = len(fig.data) - 1
        def builder(size: int) -> go.Scatter3d:
            return go.Scatter3d(x=[tx], y=[ty], z=[tz], mode="markers",
                                marker=dict(size=size, color=highlight_color, symbol="diamond", line=dict(width=3, color="#ffffff")),
                                name=f"Target: {target_label}", hoverinfo="skip")
        fig.update_layout(scene=dict(xaxis_title="Days Since Last Purchase (σ)", yaxis_title="Total Purchases (σ)",
                                     zaxis_title="Avg Order Value (σ)", bgcolor="rgba(0,0,0,0)"),
                          legend=dict(bgcolor="rgba(15,23,42,0.6)"), margin=dict(t=150, b=10, l=10, r=10),
                          height=560, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e7eb"))
    else:
        fig = px.scatter(viz_df, x="Days_Since_Last_Purchase", y="Total_Purchases", color="Segment",
                         color_discrete_map=SEGMENT_COLORS, opacity=0.60,
                         title="K-Means Customer Segments (Standardized Feature Space)")
        fig.add_trace(go.Scatter(x=[tx], y=[ty], mode="markers", marker=dict(size=44, color=highlight_color, opacity=0.30),
                                 name="Target Halo", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[tx], y=[ty], mode="markers+text", marker=dict(size=20, color=highlight_color, symbol="star", line=dict(width=2, color="#ffffff")),
                                 text=[target_label], textposition="top center", textfont=dict(size=11, color="#ffffff"),
                                 name=f"Target: {target_label}",
                                 hovertemplate=f"<b>{target_label}</b><br>Days since last purchase: %{{x:.2f}}σ<br>Total purchases: %{{y:.2f}}σ<extra></extra>"))
        highlight_idx = len(fig.data) - 1
        def builder(size: int) -> go.Scatter:
            return go.Scatter(x=[tx], y=[ty], mode="markers+text",
                              marker=dict(size=size, color=highlight_color, symbol="star", line=dict(width=2, color="#ffffff")),
                              text=[target_label], textposition="top center", textfont=dict(size=11, color="#ffffff"),
                              name=f"Target: {target_label}", hoverinfo="skip")
        fig.update_layout(xaxis_title="Days Since Last Purchase (σ)", yaxis_title="Total Purchases (σ)",
                          legend=dict(bgcolor="rgba(15,23,42,0.6)"), margin=dict(t=150, b=10, l=10, r=10),
                          height=560, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)", font=dict(color="#e5e7eb"))

    _attach_pulse_animation(fig, highlight_idx, builder)
    return fig

# =====================================================================
# 7. SIMULATION ENGINE (refactored to accept optional status container)
# =====================================================================
def execute_simulation(target_features: Dict[str, float], input_signature: Tuple,
                       display_name: str, bundle: Dict[str, Any], delay: float,
                       log_box, progress_slot, stage_slot, status=None) -> Dict[str, Any]:
    """Stage-by-stage ML pipeline with live logging, progress, and optional st.status() updates."""
    log_lines: List[str] = []
    t0 = time.time()

    def log(msg: str) -> None:
        log_lines.append(f"{_elapsed(t0)} {msg}")
        log_box.code("\n".join(log_lines), language="text")
        time.sleep(delay)

    def stage(progress: float, label: str) -> None:
        progress_slot.progress(min(progress, 1.0))
        stage_slot.markdown(f"**{label}**")
        if status is not None:
            status.update(label=label, state="running", expanded=True)
        time.sleep(delay * 0.5)

    # Stage 0
    stage(0.05, "⏳ Stage 0 · Bootstrapping pipeline runtime…")
    log("[INIT] Launching ML inference runtime (scikit-learn backend)…")
    log(f"[INIT] Target customer: {display_name}")

    # Stage 1
    stage(0.20, "⚙️ Stage 1 · Preprocessing & scaling features…")
    log("[STAGE 1] Validating input schema against training features…")
    raw_values = {col: float(target_features[col]) for col in FEATURE_COLS}
    n_missing = int(np.isnan(np.asarray(list(raw_values.values()), dtype=float)).sum())
    if n_missing:
        log(f"[STAGE 1] Detected {n_missing} missing value(s) → median imputation…")
    else:
        log("[STAGE 1] Missing-value scan clean (0 nulls) → imputation skipped")
    log("[STAGE 1] Cleaning data → imputing missing values with medians…")
    filled_values = pl.impute_missing_features(raw_values, bundle.get("feature_medians", {}))
    scaler = bundle["scaler"]
    target_df = pd.DataFrame([[filled_values[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    target_scaled = scaler.transform(target_df)
    log(f"[STAGE 1] Scaling features... mean={scaler.mean_.round(2).tolist()} scale={scaler.scale_.round(2).tolist()}")
    log(f"[STAGE 1] Scaled vector = {target_scaled.flatten().round(3).tolist()}")
    if status is not None:
        status.update(label="✅ Stage 1 · Preprocessing complete", state="complete")

    # Stage 2
    stage(0.45, "🔍 Stage 2 · Assigning customer segment (K-Means k=3)…")
    log("[STAGE 2] Loading fitted KMeans(n_clusters=3) model…")
    target_cluster_scaled = target_scaled[:, _cluster_col_indices(bundle)]
    cluster_id = int(bundle["kmeans"].predict(target_cluster_scaled)[0])
    cluster_label = CLUSTER_LABELS.get(cluster_id, "Unknown Segment")
    log(f"[STAGE 2] Cluster Assigned: {cluster_label} (id={cluster_id})")
    if status is not None:
        status.update(label="✅ Stage 2 · Clustering complete", state="complete")

    # Stage 3
    stage(0.68, "🚀 Stage 3 · Parallel inference (Classification ∥ Regression)…")
    log("[STAGE 3] Dispatching parallel inference jobs…")
    churn_probability = float(bundle["logistic_model"].predict_proba(target_scaled)[0, 1])
    log(f"[STAGE 3] ↳ LogisticRegression → churn_probability = {churn_probability:.4f}")
    predicted_ltv = float(bundle["ridge_model"].predict(target_scaled)[0])
    log(f"[STAGE 3] ↳ Ridge Regression → predicted_total_ltv = {_fmt_money(predicted_ltv)}")
    if status is not None:
        status.update(label="✅ Stage 3 · Inference complete", state="complete")

    # Stage 4
    stage(0.86, "📊 Stage 4 · Retrieving model evaluation metrics…")
    churn_metrics = bundle["metrics"]["churn"]
    ltv_metrics = bundle["metrics"]["ltv"]
    log(f"[EVAL] Churn model → ROC-AUC={churn_metrics['roc_auc']:.4f} | Accuracy={churn_metrics['accuracy']:.4f}")
    log(f"[EVAL] Confusion Matrix [[TN={churn_metrics['tn']}, FP={churn_metrics['fp']}], [FN={churn_metrics['fn']}, TP={churn_metrics['tp']}]]")
    log(f"[EVAL] LTV model → RMSE={_fmt_money(ltv_metrics['rmse'])} | MAE={_fmt_money(ltv_metrics['mae'])}")

    # Stage 5
    stage(0.96, "🎯 Stage 5 · Applying business intervention rules…")
    log(f"[STAGE 4] Evaluating rules: churn>{CHURN_THRESHOLD:.2f} + segment…")
    decision = pl.apply_decision_rules(churn_probability, cluster_id)
    log(f"[STAGE 4] Rule fired → {decision['action']}")

    # Done
    progress_slot.progress(1.0)
    stage_slot.markdown("✅ **Pipeline complete — all 4 stages executed successfully**")
    total_elapsed = _elapsed(t0)
    log(f"[DONE] Simulation finished in {total_elapsed}")
    if status is not None:
        status.update(label=f"✅ Pipeline Complete — {total_elapsed}", state="complete", expanded=True)

    return {
        "display_name": display_name,
        "input_signature": input_signature,
        "target_features": dict(target_features),
        "target_scaled": target_scaled,
        "target_cluster_scaled": target_cluster_scaled,
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "churn_probability": churn_probability,
        "predicted_ltv": predicted_ltv,
        "decision": decision,
        "log_lines": list(log_lines),
        "elapsed": total_elapsed,
    }

# =====================================================================
# 8. UI COMPONENTS
# =====================================================================
def _sample_picker(data: pd.DataFrame, test_indices: Optional[pd.Index]) -> Tuple:
    """Returns (target_features, input_signature, display_name) for sample mode."""
    if test_indices is not None and len(list(test_indices)) > 0:
        pool = data.iloc[sorted(int(i) for i in test_indices)]
    else:
        pool = data
    pool_ids = sorted(pool["Customer_ID"].astype(int).tolist())
    chosen_id = st.selectbox("Customer ID", options=pool_ids, index=0,
                             help="Hold-out test customers — never used to fit the models.")
    row = pool.loc[pool["Customer_ID"].astype(int) == chosen_id].iloc[0]
    target_features = {col: float(row[col]) for col in FEATURE_COLS}
    input_signature = ("sample", int(chosen_id))
    display_name = f"Customer #{int(chosen_id)} (test set)"
    st.session_state["_current_row"] = row.to_dict()
    return target_features, input_signature, display_name


def _manual_sliders() -> Tuple:
    """Returns (target_features, input_signature, display_name) for manual mode."""
    age = st.slider("Age", 18, 70, 35, help="Customer age in years.")
    days_since = st.slider("Days Since Last Purchase", 1, 365, 90,
                           help="Dormancy period — higher means more at-risk.")
    total_purchases = st.slider("Total Purchases", 1, 50, 10)
    avg_order_value = st.slider("Avg Order Value ($)", 20.0, 500.0, 95.0, step=5.0)
    support_tickets = st.slider("Support Tickets", 0, 10, 1)
    target_features = {
        "Age": float(age),
        "Days_Since_Last_Purchase": float(days_since),
        "Total_Purchases": float(total_purchases),
        "Avg_Order_Value": float(avg_order_value),
        "Support_Tickets": float(support_tickets),
    }
    input_signature = ("manual", age, days_since, int(total_purchases),
                       round(avg_order_value, 2), int(support_tickets))
    display_name = "Manual Slider Profile"
    st.session_state["_current_row"] = {
        "Age": age, "Total_Purchases": total_purchases,
        "Avg_Order_Value": avg_order_value, "Support_Tickets": support_tickets,
        "Days_Since_Last_Purchase": days_since,
    }
    return target_features, input_signature, display_name


def render_target_profile_card(target_features: Dict[str, float],
                               display_name: str,
                               sim: Optional[Dict[str, Any]]) -> None:
    """Render the Target Profile card with high-density badge summary."""
    st.markdown("##### 🎯 Target Profile")
    st.caption(display_name)

    # Compact attribute grid using st.columns with strict uniform height
    row1 = st.columns(2, gap="small")
    with row1[0]:
        st.metric("Age", _fmt_feature(target_features.get("Age", float("nan"))))
    with row1[1]:
        st.metric("Purchases", _fmt_feature(target_features.get("Total_Purchases", float("nan"))))

    row2 = st.columns(2, gap="small")
    with row2[0]:
        st.metric("Avg Order", _fmt_feature(target_features.get("Avg_Order_Value", float("nan")), " $"))
    with row2[1]:
        st.metric("Tickets", _fmt_feature(target_features.get("Support_Tickets", float("nan"))))

    row3 = st.columns(2, gap="small")
    with row3[0]:
        st.metric("Days Since", _fmt_feature(target_features.get("Days_Since_Last_Purchase", float("nan"))))
    with row3[1]:
        st.metric("Churned?", "—" if sim is None else ("Yes" if float(target_features.get("Churned", 0)) == 1 else "No"))

    # Decision result badge (if simulation ran)
    if sim is not None:
        dec = sim["decision"]
        if dec["severity"] == "high":
            badge_text = "🔴 HIGH RISK — INTERVENTION REQUIRED"
        else:
            badge_text = "🟢 LOW RISK — NO ACTION NEEDED"
        st.caption(badge_text)
        # Segment badge
        seg_color = SEGMENT_COLORS.get(sim["cluster_label"], "#9a9a9a")
        st.caption(f"Segment: <b><span style='color:{seg_color}'>{sim['cluster_label']}</span></b>", unsafe_allow_html=True)


def render_terminal_log(sim: Optional[Dict[str, Any]], log_box) -> None:
    """Render the terminal log card with monospace text stream."""
    if sim is not None:
        st.markdown(
            f'<div class="terminal-log terminal-scroll"><pre style="max-height:350px;">{chr(10).join(sim["log_lines"])}</pre></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="terminal-log terminal-scroll"><pre style="color:#9a9a9a;max-height:350px;">'
            'Initializing pipeline runtime…\n'
            'Awaiting target customer input.\n'
            'Click "⚡ Trigger Pipeline Model" to begin.</pre></div>',
            unsafe_allow_html=True,
        )


def render_aggregate_analytics(bundle: Dict[str, Any], data: pd.DataFrame) -> None:
    """Render the Aggregate Analytics view."""
    churn_m = bundle["metrics"]["churn"]
    ltv_m = bundle["metrics"]["ltv"]
    insights = bundle.get("insights", {})

    # Segment distribution + model metrics
    left_col, right_col = st.columns([1.5, 1], gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown("##### 📊 Segment Distribution")
            labels = bundle["predictions"]["cluster_labels"]
            seg_counts = pd.Series(labels).value_counts()
            seg_df = pd.DataFrame({
                "Segment": [CLUSTER_LABELS.get(k, "Unknown") for k in seg_counts.index],
                "Customers": seg_counts.values,
            })
            fig = px.bar(seg_df, x="Customers", y="Segment", orientation="h",
                         color="Segment", color_discrete_map=SEGMENT_COLORS,
                         title=None, labels={"Customers": "Count"})
            fig.update_layout(height=300, margin=dict(t=20, b=10, l=10, r=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#e5e7eb"), showlegend=False,
                              xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig, width="stretch")

    with right_col:
        with st.container(border=True):
            st.markdown("##### 📈 Model Performance")
            with st.expander("Churn Classifier (LogisticRegression)", expanded=True):
                cm = np.array([[churn_m["tn"], churn_m["fp"]], [churn_m["fn"], churn_m["tp"]]])
                cm_fig = px.imshow(cm, x=["Pred: Active", "Pred: Churned"],
                                   y=["Actual: Active", "Actual: Churned"],
                                   color_continuous_scale="Blues", text_auto=True,
                                   title="Confusion Matrix")
                cm_fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                                     font=dict(color="#e5e7eb"), margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(cm_fig, width="stretch")
            with st.expander("LTV Regressor (Ridge)", expanded=True):
                st.caption(f"RMSE: {_fmt_money(ltv_m['rmse'])} · MAE: {_fmt_money(ltv_m['mae'])}")

    # Segment insights table
    st.markdown("##### 📋 Segment Business Summary")
    insights_df = pd.DataFrame(insights).T
    if not insights_df.empty:
        insights_df = insights_df.reset_index().rename(columns={"index": "Segment"})
        display_df = insights_df.copy()
        display_df["churn_rate"] = (display_df["churn_rate"] * 100).round(1).astype(str) + "%"
        for col in ["avg_ltv", "avg_age", "avg_purchases", "avg_aov", "avg_support_tickets"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        st.dataframe(display_df, width="stretch", hide_index=True)


def render_pipeline_config(bundle: Dict[str, Any]) -> None:
    """Render the Pipeline Config view."""
    st.markdown("##### ⚙️ Pipeline Configuration")
    st.caption("Trained model hyperparameters and pipeline parameters.")

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown("**Preprocessing Stage**")
            st.markdown(f"- Imputation: Median (feature medians: {bundle.get('feature_medians', {})} )")
            st.markdown(f"- Scaler: StandardScaler (mean → scale)")
            means = bundle["scaler"].mean_.round(3)
            scales = bundle["scaler"].scale_.round(3)
            st.markdown(f"- Mean vector: `{means.tolist()}`")
            st.markdown(f"- Scale vector: `{scales.tolist()}`")

    with right_col:
        with st.container(border=True):
            st.markdown("**Clustering — KMeans(k=3)**")
            st.markdown(f"- Feature subspace: {list(bundle.get('cluster_feature_cols', PLOT_FEATURES))}")
            st.markdown(f"- n_clusters: 3")
            st.markdown(f"- n_init: 10")
            st.markdown(f"- random_state: 42")
            centroids = bundle["kmeans"].cluster_centers_
            st.markdown(f"- Centroid count: {len(centroids)}")

    with st.container(border=True):
        st.markdown("**Supervised Models**")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**LogisticRegression (Churn)**")
            st.markdown(f"- C: 1.0")
            st.markdown(f"- solver: lbfgs")
            st.markdown(f"- n_iter: {bundle['logistic_model'].n_iter_}")
        with m2:
            st.markdown("**Ridge (LTV)**")
            st.markdown(f"- alpha: 1.0")
            st.markdown(f"- fit_intercept: True")
        st.markdown(f"- Churn threshold: {CHURN_THRESHOLD:.2f}")

    # Decision rules card
    with st.container(border=True):
        st.markdown("**Decision Rules (Intervention Engine)**")
        st.markdown(
            "| Condition | Action | Severity |\n"
            "|---|---|---|\n"
            f"| Churn > {CHURN_THRESHOLD:.2f} AND Segment == At-Risk | 🎟️ 15% Discount Voucher | HIGH |\n"
            f"| Churn > {CHURN_THRESHOLD:.2f} AND Segment == VIP | 👑 Priority VIP Outreach | HIGH |\n"
            "| Else | ✅ No Action Required | LOW |"
        )

    # Dataset info
    with st.container(border=True):
        st.markdown("**Dataset Artifact**")
        st.markdown(f"- Source: `data/synthetic_customers.csv`")
        st.markdown(f"- Model bundle: `models/pipeline_models.joblib`")
        st.markdown(f"- Rows: {len(bundle['data']):,} · Cols: {len(bundle['data'].columns)}")

    # Feature importance (if available)
    try:
        coef = bundle["logistic_model"].coef_[0]
        feat_imp = pd.DataFrame({"feature": FEATURE_COLS, "coef": coef})
        feat_imp = feat_imp.reindex(feat_imp.abs().sort_values(ascending=False).index)
        fi_fig = px.bar(feat_imp, x="feature", y="coef", orientation="h",
                        title="Churn Model Feature Coefficients")
        fi_fig.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10),
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font=dict(color="#e5e7eb"), showlegend=False,
                             xaxis=dict(showgrid=False))
        st.plotly_chart(fi_fig, width="stretch")
    except Exception:
        pass


# =====================================================================
# 9. MAIN APPLICATION
# =====================================================================
def main() -> None:
    """Entry point: wires up the Streamlit dashboard."""

    # ------------------------------------------------------------------
    # Load data & train (cached) pipeline bundle
    # ------------------------------------------------------------------
    try:
        data = load_or_generate_data()
        with st.spinner("Training ML pipeline (first run only)…"):
            bundle = train_pipeline_bundle()
    except FileNotFoundError as exc:
        st.error(f"❌ Data file missing: {exc}")
        st.stop()
        return
    except Exception as exc:
        st.error(f"❌ Pipeline initialization failed: {exc}")
        st.exception(exc)
        st.stop()
        return

    clean_data_df: pd.DataFrame = bundle["data"]
    historical_avg_ltv: float = float(clean_data_df["Total_LTV"].mean())
    baseline_churn_pct: float = float(clean_data_df["Churned"].mean() * 100.0)

    # ------------------------------------------------------------------
    # SESSION STATE INIT
    # ------------------------------------------------------------------
    if "sim_result" not in st.session_state:
        st.session_state["sim_result"] = None
    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "sample"
    if "sim_speed" not in st.session_state:
        st.session_state["sim_speed"] = "Normal 🚀"
    if "run_flag" not in st.session_state:
        st.session_state["run_flag"] = False
    if "config_mode" not in st.session_state:
        st.session_state["config_mode"] = False

    # ------------------------------------------------------------------
    # TOP HEADER ROW
    # ------------------------------------------------------------------
    header_center, header_right = st.columns([2.4, 1.2], gap="medium", vertical_alignment="center")

    with header_center:
        global_mode = st.segmented_control(
            "Global Mode",
            options=["🎯 Live Simulation", "📊 Aggregate Analytics", "⚙️ Pipeline Config"],
            default="🎯 Live Simulation",
            key="global_mode",
        )

    with header_right:
        st.markdown("<div style='font-size:0.7rem;color:#9a9a9a;margin-bottom:4px;'>PIPELINE ENGINE</div>", unsafe_allow_html=True)
        speed_label = st.selectbox(
            "Simulation Speed",
            options=list(SIM_SPEEDS.keys()),
            index=1,
            help="Controls the delay between live terminal log events.",
        )
        delay = SIM_SPEEDS[speed_label]

    # ------------------------------------------------------------------
    # ROUTE: GLOBAL MODE
    # ------------------------------------------------------------------
    if global_mode.startswith("📊"):
        # === AGGREGATE ANALYTICS ===
        st.write("")
        with st.container(border=True):
            render_aggregate_analytics(bundle, data)
        return

    if global_mode.startswith("⚙️"):
        # === PIPELINE CONFIG ===
        st.write("")
        with st.container(border=True):
            render_pipeline_config(bundle)
        return

    # === LIVE SIMULATION MODE ===

    # ------------------------------------------------------------------
    # ACTION BAR — popover & trigger button share one row
    # ------------------------------------------------------------------
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        with st.popover("🎛️ Configure Target Customer"):
            input_mode = st.radio(
                "Input Mode",
                options=["🎯 Test Set Sample", "🎚️ Manual Slider Input"],
                index=0,
                help="Choose an existing customer from the test pool or craft a custom profile.",
            )
            if input_mode.startswith("🎯"):
                target_features, input_signature, display_name = _sample_picker(
                    data, test_indices=bundle.get("test_indices")
                )
            else:
                target_features, input_signature, display_name = _manual_sliders()
            st.caption("All five inputs feed the models: K-Means segments on Days / Purchases / AOV; churn & LTV also weigh Age and Support Tickets.")
            st.dataframe(data.head(8), width="stretch", hide_index=True)
            st.caption(f"Shape: {data.shape[0]} rows × {data.shape[1]} cols · Churn: {data['Churned'].mean()*100:.1f}%")

    with c2:
        run_clicked = st.button(
            "⚡ Trigger Pipeline Model",
            type="primary",
            width="stretch",
            help="Animates every backend stage with live logs, progress bars and charts.",
        )

    # ------------------------------------------------------------------
    # TRIPLE-COLUMN CARD GRID
    # ------------------------------------------------------------------
    col1, col2, col3 = st.columns([1, 1.2, 1.3], gap="medium")
    st.write("")  # space before columns

    # === COLUMN 1: Target Profile Card ===
    with col1:
        with st.container(border=True):
            st.markdown("<div style='font-size:0.7rem;color:#9a9a9a;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase;'>Profile</div>", unsafe_allow_html=True)
            st.write("")  # space inside Profile card
            render_target_profile_card(target_features, display_name, st.session_state.get("sim_result"))

    # === COLUMN 2: Interactive Action & Pipeline Feed ===
    with col2:
        with st.container(border=True):
            st.markdown("<div style='font-size:0.7rem;color:#9a9a9a;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase;'>Pipeline Feed</div>", unsafe_allow_html=True)
            st.markdown("##### ⚡ Pipeline Execution Feed")
            st.write("")  # space inside Feed card
            log_box = st.empty()
            progress_slot = st.empty()
            stage_slot = st.empty()

            if run_clicked:
                # Reset and run
                st.session_state["sim_result"] = None
                status = st.status("⏳ Pipeline Starting…", state="running", expanded=True)
                try:
                    result = execute_simulation(
                        target_features=target_features,
                        input_signature=input_signature,
                        display_name=display_name,
                        bundle=bundle,
                        delay=delay,
                        log_box=log_box,
                        progress_slot=progress_slot,
                        stage_slot=stage_slot,
                        status=status,
                    )
                    st.session_state["sim_result"] = result
                except Exception as exc:
                    status.update(label=f"❌ Error: {exc}", state="error", expanded=True)
                    st.error(f"❌ Simulation failed: {exc}")
                    st.exception(exc)
            elif st.session_state.get("sim_result") is not None:
                # Restore final log on rerun
                prior = st.session_state["sim_result"]
                log_box.code("\n".join(prior["log_lines"]), language="text")
                progress_slot.progress(1.0)
                stage_slot.markdown("✅ **Pipeline complete**")
                st.status(f"✅ Pipeline Complete — {prior['elapsed']}", state="complete", expanded=True)
            else:
                # Idle state
                st.status("Ready — configure inputs and click Trigger", state="complete", expanded=False)

    # === Column 3: Real-Time Terminal Log ===
    with col3:
        with st.container(border=True):
            st.markdown("<div style='font-size:0.7rem;color:#9a9a9a;letter-spacing:1px;margin-bottom:20px;text-transform:uppercase;'>Terminal</div>", unsafe_allow_html=True)
            render_terminal_log(st.session_state.get("sim_result"), None)

    st.write("")  # spacer

    # ------------------------------------------------------------------
    # VISUAL INSIGHTS FOOTER
    # ------------------------------------------------------------------
    st.write("")
    sim: Optional[Dict[str, Any]] = st.session_state.get("sim_result")

    # Stale-input detection
    if sim is not None and sim["input_signature"] != input_signature:
        st.warning(
            "⚠️ Inputs changed since the last run — displaying previous results. "
            "Press **⚡ Trigger Pipeline Model** to refresh."
        )

    st.write("")  # space before segmentation container
    # === Step 1: K-Means Segmentation ===
    with st.container(border=True):
        st.markdown("##### 🗺️ Behavioral Segmentation Space")
        st.caption("K-Means customer clusters with pulsing target highlight. Press ▶ Pulse Target for animation.")

        st.markdown('<div style="margin-bottom:8px;"></div>', unsafe_allow_html=True)
        view_col, _ = st.columns([1, 4])
        with view_col:
            view = st.radio("Plot view", options=["2D", "3D"], index=0, horizontal=True)
        st.write("")  # spacer between plot controls and chart

        if sim is not None:
            target_cluster_scaled = sim.get("target_cluster_scaled", sim["target_scaled"])
            cluster_id = sim["cluster_id"]
            target_label = sim["display_name"]
        else:
            idle_target = bundle["scaler"].transform(
                clean_data_df[FEATURE_COLS].head(1)
            )[:, _cluster_col_indices(bundle)]
            target_cluster_scaled = idle_target
            cluster_id = int(bundle["predictions"]["cluster_labels"][0])
            target_label = "Preview"

        cluster_fig = build_cluster_figure(
            bundle=bundle,
            target_cluster_scaled=target_cluster_scaled,
            cluster_id=cluster_id,
            target_label=target_label,
            view=view,
        )
        st.plotly_chart(cluster_fig, width="stretch", key="cluster_chart")
        st.write("")  # spacer between chart and segment summary

        # Segment summary strip
        if sim is not None:
            insights = bundle.get("insights", {})
            seg1, seg2, seg3 = st.columns(3)
            seg1.metric("Assigned Segment", sim["cluster_label"])
            seg2.metric("Cluster ID", sim["cluster_id"])
            seg3.metric(
                "Segment Avg LTV",
                _fmt_money(insights.get(sim["cluster_label"], {}).get("avg_ltv", 0.0)),
            )

    st.write("")  # spacer before Segment Insights table
    # Segment insights table
    insights = bundle.get("insights", {})
    st.markdown("##### 📋 Segment Insights (business summary)")
    insights_df = pd.DataFrame(insights).T
    if not insights_df.empty:
        insights_df = insights_df.reset_index().rename(columns={"index": "Segment"})
        display_df = insights_df.copy()
        display_df["churn_rate"] = (display_df["churn_rate"] * 100).round(1).astype(str) + "%"
        for col in ["avg_ltv", "avg_age", "avg_purchases", "avg_aov", "avg_support_tickets"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        st.dataframe(display_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()

