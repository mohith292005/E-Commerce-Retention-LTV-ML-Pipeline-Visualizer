# E-Commerce Customer Retention & LTV — ML Pipeline Visualizer

An interactive Streamlit dashboard that **animates the execution of an
end-to-end Scikit-Learn pipeline** for e-commerce customer retention and
lifetime-value prediction — timestamped terminal logs, staged progress bars,
Plotly cluster maps with a pulsing target marker, a churn gauge chart, an LTV
metric card and an automated decision banner.

## Run it

```bash
pip install -r requirements.txt
python data_generator.py     # optional: regenerate data/synthetic_customers.csv
python pipeline.py           # optional: retrain models/pipeline_models.joblib
streamlit run app.py
```

The app is self-contained: on first launch it generates the dataset, trains
the pipeline and persists the model bundle automatically.

## Architecture

| Stage | What happens | Where |
|---|---|---|
| 0. Data | 1,000 synthetic customers; ~25% churn (quantile-split risk score); ~5% NaNs injected into feature columns only | `data_generator.py` |
| 1. Preprocess | Median imputation (training medians cached for train/serve parity) + `StandardScaler` on all 5 features | `pipeline.py` |
| 2. Segment | `KMeans(n_clusters=3)` on the behavioural subspace (*Days since last purchase, Total purchases, Avg order value*); centroids are permuted so IDs match the business labels: **0 = At-Risk Bargain Hunters, 1 = Consistent Mid-Tier, 2 = VIP High Spenders** | `pipeline.py` |
| 3. Infer (parallel) | `LogisticRegression` -> churn probability (ROC-AUC ~ 0.79) · `Ridge` -> continuous Total LTV (RMSE ~ $145) | `pipeline.py` |
| 4. Decide | `churn > 0.70` + At-Risk -> *15% discount voucher*; `churn > 0.70` + VIP -> *priority VIP outreach*; else -> *no action* | `pipeline.py` |

Supervised models consume **all five** features (Age, Days since last
purchase, Total purchases, Avg order value, Support tickets); K-Means and the
scatter plots use the three-feature behavioural subspace.

## Dashboard layout

- **Top section — Pipeline Simulator & Live Controls**: sidebar inputs
  (hold-out **test-set** sample picker or manual sliders), simulation-speed
  control, **Run Pipeline Simulation** button, live timestamped terminal
  log `[MM:SS.mm]` and staged progress bars (`time.sleep`-paced).
- **Main section — Visualization Canvas**:
  1. Plotly **2D/3D** K-Means cluster scatter with the target customer
     highlighted and **pulsing** (Pulse Target animation).
  2. Parallel inference cards: churn-probability **gauge** (green/yellow/red)
     and **Predicted LTV vs Historical Average** metric card + comparison bar.
  3. Highlighted **decision banner** from the intervention engine, plus a
     hold-out evaluation expander (ROC-AUC, confusion matrix, RMSE, MAE).

## Project structure

```
├── app.py                  # Streamlit dashboard: UI, live logs, Plotly visuals, simulation state
├── pipeline.py             # Preprocessing, K-Means, LogisticRegression, Ridge, decision rules
├── data_generator.py       # Synthetic customer dataset -> data/synthetic_customers.csv
├── data/                   # Generated dataset artifact
├── models/                 # Persisted model bundle (joblib)
├── test_app_smoke.py       # AppTest: idle render, full simulation, canvas assertions
├── test_edge_cases.py      # AppTest: NaN-customer imputation + manual-slider simulation
├── test_decision_rules.py  # Unit checks: decision rules, medians, imputation error paths
├── requirements.txt
└── README.md
```

## Tests

```bash
python test_decision_rules.py   # fast unit checks
python test_app_smoke.py        # full AppTest smoke run (~1 min)
python test_edge_cases.py       # edge cases (~1 min)
```
