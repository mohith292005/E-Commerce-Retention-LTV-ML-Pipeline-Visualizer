"""
pipeline.py

End-to-end Machine Learning pipeline functions for E-Commerce Customer Retention
and Lifetime Value (LTV) Prediction.

Stages:
1. Feature Preprocessing & Scaling
2. Unsupervised Clustering (K-Means - Customer Segmentation)
3. Supervised Learning Inference:
   - Classification: Logistic Regression for Churn Probability
   - Regression: Ridge Regression for Total LTV Prediction
4. Business Logic & Intervention Engine

Provides functions to:
- Preprocess data (imputation, scaling)
- Train and evaluate models
- Save/load fitted models and scalers
- Apply decision rules for interventions
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
)


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

# All raw features used for scaling AND supervised learning (classification &
# regression). Using every available signal means the Age and Support_Tickets
# inputs in the dashboard genuinely influence the churn/LTV predictions.
MODEL_FEATURES = [
    "Age",
    "Days_Since_Last_Purchase",
    "Total_Purchases",
    "Avg_Order_Value",
    "Support_Tickets",
]

# Behavioural subset used for K-Means segmentation (per architecture spec):
# clustering happens in the Days / Purchases / Avg-Order-Value subspace only.
CONTINUOUS_FEATURES = [
    "Days_Since_Last_Purchase",
    "Total_Purchases",
    "Avg_Order_Value",
]
CLUSTER_FEATURES = CONTINUOUS_FEATURES  # explicit alias

# Default feature set for select_features() = full supervised feature set
DEFAULT_FEATURES = MODEL_FEATURES

TARGET_COLUMN_CHURN = "Churned"
TARGET_COLUMN_LTV = "Total_LTV"

# Model hyperparameters
KMEANS_N_CLUSTERS = 3
LOGISTIC_REGRESSION_C = 1.0
RIDGE_ALPHA = 1.0

# Cluster label mapping
CLUSTER_LABELS = {
    0: "At-Risk Bargain Hunters",
    1: "Consistent Mid-Tier",
    2: "VIP High Spenders",
}

# Decision rule thresholds
CHURN_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Stage 1: Feature Preprocessing & Scaling
# ---------------------------------------------------------------------------


def load_data(filepath: str = "data/synthetic_customers.csv") -> pd.DataFrame:
    """
    Load customer data from CSV.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing synthetic customer data.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[PIPELINE] Loaded {len(df)} records from {filepath}")
    return df


def compute_feature_medians(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
) -> Dict[str, float]:
    """
    Compute per-column median imputation values from a raw DataFrame.

    Medians are derived once from the training data and reused both for
    fitting the dataset and for single-record inference in the dashboard,
    guaranteeing train/serve consistency (no leakage, no drift).

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    feature_cols : list, optional
        Columns to compute medians for. Defaults to ``MODEL_FEATURES``.

    Returns
    -------
    Dict[str, float]
        Mapping of column name -> median value.
    """
    if feature_cols is None:
        feature_cols = MODEL_FEATURES

    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Cannot compute medians; columns missing: {missing}")

    return {col: float(df[col].median()) for col in feature_cols}


def impute_missing_features(
    feature_values: Dict[str, float],
    medians: Dict[str, float],
) -> Dict[str, float]:
    """
    Impute NaN entries in a single customer's feature dict using training medians.

    Parameters
    ----------
    feature_values : Dict[str, float]
        Raw feature mapping; values may be NaN (missing).
    medians : Dict[str, float]
        Training medians (see :func:`compute_feature_medians`).

    Returns
    -------
    Dict[str, float]
        Copy of the mapping with every NaN replaced by its column median.

    Raises
    ------
    ValueError
        If a median is unavailable for a column that needs imputation.
    """
    filled: Dict[str, float] = {}
    for col, value in feature_values.items():
        v = float(value)
        if np.isnan(v):
            if col not in medians or medians.get(col) is None:
                raise ValueError(
                    f"Missing value for '{col}' but no training median is available."
                )
            v = float(medians[col])
        filled[col] = v
    return filled


def clean_data(
    df: pd.DataFrame,
    medians: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Clean the customer DataFrame by handling missing values.

    - Numeric columns: Impute missing values with the median.
    - Supervision targets (Churned / Total_LTV) are left untouched.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    medians : Dict[str, float], optional
        Pre-computed training medians. When omitted they are computed from
        ``df`` itself (equivalent for a single-pass fit).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no missing values in feature columns.
    """
    print("[PIPELINE] Cleaning data and imputing missing values with medians...")

    df_clean = df.copy()

    # Identify numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    # Never impute the identifier column; targets stay as authored
    id_col = "Customer_ID"
    feature_cols = [c for c in numeric_cols if c not in [id_col]]

    if medians is None:
        medians = {
            col: float(df_clean[col].median())
            for col in feature_cols
            if df_clean[col].isnull().any()
        }

    # Impute missing values with median for each feature column
    # (plain assignment — chained in-place fillna is a no-op under pandas Copy-on-Write)
    for col in feature_cols:
        if df_clean[col].isnull().any() and col in medians:
            df_clean[col] = df_clean[col].fillna(float(medians[col]))

    # Double-check: no NaNs remain in feature columns
    remaining_nans = df_clean[feature_cols].isnull().sum().sum()
    if remaining_nans > 0:
        print(
            f"[WARNING] {remaining_nans} NaN values remain after imputation."
        )

    print(f"[PIPELINE] Cleaned data shape: {df_clean.shape}")
    return df_clean


def select_features(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
) -> Tuple[pd.DataFrame, list]:
    """
    Select the feature matrix from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    feature_cols : list, optional
        List of feature column names. If None, uses MODEL_FEATURES
        (the full supervised feature set).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    feature_cols : list
        List of selected feature column names.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    # Verify all requested columns exist
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Feature columns not found in DataFrame: {missing}")

    X = df[feature_cols].copy()
    print(f"[PIPELINE] Selected features: {feature_cols}")
    return X, feature_cols


def scale_features(
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Apply StandardScaler to normalize continuous features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (un-scaled).
    scaler : StandardScaler, optional
        Pre-fitted scaler. If fit=True, a new scaler is fitted on X.
    fit : bool
        Whether to fit the scaler on the data (True) or just transform (False).

    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature matrix.
    scaler : StandardScaler
        The scaler object (fitted if fit=True, otherwise provided).
    """
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(
            f"[PIPELINE] Fitted StandardScaler on {X.shape[0]} samples, "
            f"{X.shape[1]} features. Mean/std computed per feature."
        )
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False.")
        X_scaled = scaler.transform(X)
        print(
            f"[PIPELINE] Transformed features using existing scaler."
        )

    return X_scaled, scaler


# ---------------------------------------------------------------------------
# Stage 2: Unsupervised Clustering (K-Means)
# ---------------------------------------------------------------------------


def train_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = KMEANS_N_CLUSTERS,
    random_state: int = 42,
) -> KMeans:
    """
    Train K-Means clustering model on scaled features.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix — the CLUSTER_FEATURES subspace
        (Days_Since_Last_Purchase, Total_Purchases, Avg_Order_Value).
    n_clusters : int
        Number of clusters (default: 3).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    KMeans
        Fitted K-Means model.
    """
    print(
        f"[PIPELINE] Training K-Means clustering (n_clusters={n_clusters})..."
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
        verbose=0,
    )
    kmeans.fit(X_scaled)
    print(
        f"[PIPELINE] K-Means training complete. "
        f"Inertia: {kmeans.inertia_:.2f}"
    )
    # KMeans assigns cluster IDs arbitrarily; align them with the fixed
    # business definitions: 0 = At-Risk, 1 = Consistent Mid-Tier, 2 = VIP.
    _align_clusters_to_business_labels(kmeans)
    return kmeans


def _align_clusters_to_business_labels(kmeans: KMeans) -> None:
    """
    Permute fitted K-Means centroids so cluster IDs match business semantics.

    Business score per centroid (in scaled feature space, where features are
    ordered as CONTINUOUS_FEATURES = [Days_Since_Last_Purchase,
    Total_Purchases, Avg_Order_Value]):

        score = Total_Purchases + Avg_Order_Value - Days_Since_Last_Purchase

    Ranking:
        lowest  score -> cluster 0  (At-Risk Bargain Hunters)
        middle  score -> cluster 1  (Consistent Mid-Tier)
        highest score -> cluster 2  (VIP High Spenders)

    Mutates ``kmeans.cluster_centers_`` in place, so subsequent calls to
    ``predict``/``labels_`` automatically reflect the new IDs.

    Parameters
    ----------
    kmeans : KMeans
        A fitted K-Means model.
    """
    centers = kmeans.cluster_centers_
    # Feature positions (must match CONTINUOUS_FEATURES order)
    days_idx = CONTINUOUS_FEATURES.index("Days_Since_Last_Purchase")
    purchases_idx = CONTINUOUS_FEATURES.index("Total_Purchases")
    aov_idx = CONTINUOUS_FEATURES.index("Avg_Order_Value")

    business_score = (
        centers[:, purchases_idx] + centers[:, aov_idx] - centers[:, days_idx]
    )
    # ascending rank: 0 -> lowest score (worst segment)
    order = np.argsort(business_score)  # order[i] = centroid index with rank i
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))

    # Permute centers: new position i holds the centroid ranked i
    kmeans.cluster_centers_ = centers[order].copy()
    # Keep stored training labels consistent with the permutation
    if hasattr(kmeans, "labels_") and kmeans.labels_ is not None:
        kmeans.labels_ = inverse[kmeans.labels_]

    print(
        "[PIPELINE] Cluster IDs aligned to business segments "
        "(0=At-Risk, 1=Mid-Tier, 2=VIP)."
    )


def assign_clusters(
    X_scaled: np.ndarray,
    kmeans: KMeans,
) -> np.ndarray:
    """
    Assign cluster labels to each sample.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    kmeans : KMeans
        Fitted K-Means model.

    Returns
    -------
    np.ndarray
        Cluster labels (0, 1, 2).
    """
    labels = kmeans.predict(X_scaled)
    return labels


def map_cluster_labels(
    cluster_id: int,
) -> str:
    """
    Map a numeric cluster ID to a human-readable business label.

    Parameters
    ----------
    cluster_id : int
        Numeric cluster label (0, 1, or 2).

    Returns
    -------
    str
        Business segment label.
    """
    return CLUSTER_LABELS.get(cluster_id, "Unknown Segment")


def evaluate_clustering(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    kmeans: Optional[KMeans] = None,
) -> Dict[str, Any]:
    """
    Evaluate clustering quality using silhouette and Calinski-Harabasz indices.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    labels : np.ndarray
        Cluster assignments.
    kmeans : KMeans, optional
        Fitted K-Means model (used to report inertia when provided).

    Returns
    -------
    Dict[str, Any]
        Dictionary of evaluation metrics.
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    try:
        silhouette = float(silhouette_score(X_scaled, labels))
        ch_score = float(calinski_harabasz_score(X_scaled, labels))
    except Exception:
        silhouette = None
        ch_score = None

    return {
        "inertia": float(kmeans.inertia_) if kmeans is not None else None,
        "silhouette_score": silhouette,
        "calinski_harabasz_score": ch_score,
    }


# ---------------------------------------------------------------------------
# Stage 3: Parallel Supervised Learning Inference
# ---------------------------------------------------------------------------


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = LOGISTIC_REGRESSION_C,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train Logistic Regression model to predict Churn Probability.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target labels (0/1).
    C : float
        Inverse regularization strength.
    random_state : int
        Random seed.

    Returns
    -------
    LogisticRegression
        Fitted Logistic Regression model.
    """
    print(
        f"[PIPELINE] Training Logistic Regression for Churn Prediction "
        f"(C={C})..."
    )
    model = LogisticRegression(
        C=C,
        random_state=random_state,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    print("[PIPELINE] Logistic Regression training complete.")
    return model


def train_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = RIDGE_ALPHA,
    random_state: int = 42,
) -> Ridge:
    """
    Train Ridge Regression model to predict continuous Total_LTV.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target values (continuous LTV).
    alpha : float
        Regularization strength.
    random_state : int
        Random seed.

    Returns
    -------
    Ridge
        Fitted Ridge regression model.
    """
    print(
        f"[PIPELINE] Training Ridge Regression for LTV Prediction "
        f"(alpha={alpha})..."
    )
    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(X_train, y_train)
    print("[PIPELINE] Ridge Regression training complete.")
    return model


def evaluate_classification(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate the Logistic Regression classification model.

    Metrics:
    - ROC-AUC score
    - Confusion Matrix components

    Parameters
    ----------
    model : LogisticRegression
        Fitted classification model.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test true labels.

    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics.
    """
    # Predict probabilities for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    try:
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        # Raised when the hold-out set contains a single class
        roc_auc = float("nan")

    # Confusion matrix — labels=[0, 1] guarantees a 2x2 matrix even when one
    # class is absent from y_test / y_pred (otherwise .ravel() underflows).
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "roc_auc": roc_auc,
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "precision": float(tp / (tp + fp + 1e-8)),
        "recall": float(tp / (tp + fn + 1e-8)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def evaluate_regression(
    model: Ridge,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate the Ridge Regression model for LTV prediction.

    Metrics:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)

    Parameters
    ----------
    model : Ridge
        Fitted regression model.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test true target values.

    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "rmse_percentage": float(
            (rmse / np.mean(y_test) * 100) if np.mean(y_test) != 0 else 0
        ),
    }
    return metrics


def predict_churn_probability(
    model: LogisticRegression,
    X_scaled: np.ndarray,
) -> np.ndarray:
    """
    Predict churn probability (0.0 to 1.0) for given features.

    Parameters
    ----------
    model : LogisticRegression
        Fitted Logistic Regression model.
    X_scaled : np.ndarray
        Scaled feature matrix.

    Returns
    -------
    np.ndarray
        Array of churn probabilities (shape: (n_samples,)).
    """
    probs = model.predict_proba(X_scaled)[:, 1]
    return probs


def predict_ltv(
    model: Ridge,
    X_scaled: np.ndarray,
) -> np.ndarray:
    """
    Predict continuous Total_LTV for given features.

    Parameters
    ----------
    model : Ridge
        Fitted Ridge regression model.
    X_scaled : np.ndarray
        Scaled feature matrix.

    Returns
    -------
    np.ndarray
        Array of predicted LTV values.
    """
    preds = model.predict(X_scaled)
    return preds


# ---------------------------------------------------------------------------
# Stage 4: Business Logic & Intervention Engine
# ---------------------------------------------------------------------------


def apply_decision_rules(
    churn_probability: float,
    cluster_id: int,
) -> Dict[str, Any]:
    """
    Apply business logic rules to determine automated intervention actions.

    Rules (priority order):
    1. IF Churn Probability > 0.70 AND Cluster == "At-Risk" -> "Trigger Automated 15% Discount Voucher".
    2. IF Churn Probability > 0.70 AND Cluster == "VIP" -> "Flag for Priority VIP Customer Support Outreach".
    3. ELSE -> "No Action Needed (Customer Active/Low Risk)".

    Parameters
    ----------
    churn_probability : float
        Model-predicted churn probability (0.0 to 1.0).
    cluster_id : int
        Numeric cluster label (0, 1, 2).

    Returns
    -------
    Dict[str, Any]
        Decision payload with keys ``action``, ``severity``,
        ``churn_probability``, ``cluster_id`` and ``cluster_label``.
    """
    cluster_label = map_cluster_labels(cluster_id)

    rule_churn_above = churn_probability > CHURN_THRESHOLD

    if rule_churn_above and cluster_id == 0:
        action = "Trigger Automated 15% Discount Voucher"
        severity = "high"
    elif rule_churn_above and cluster_id == 2:
        action = "Flag for Priority VIP Customer Support Outreach"
        severity = "high"
    else:
        action = "No Action Needed (Customer Active/Low Risk)"
        severity = "low"

    return {
        "action": action,
        "severity": severity,
        "churn_probability": float(churn_probability),
        "cluster_id": int(cluster_id),
        "cluster_label": cluster_label,
    }


def get_business_segment_insights(
    df: pd.DataFrame,
    cluster_col: np.ndarray,
) -> Dict[str, Any]:
    """
    Generate summary insights per cluster segment for business stakeholders.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe with predictions added.
    cluster_col : np.ndarray
        Cluster assignment for each row.

    Returns
    -------
    Dict[str, Any]
        Insights per segment: churn rate, avg LTV, avg purchases, etc.
    """
    insights = {}
    df_with_cluster = df.copy()
    df_with_cluster["Cluster"] = cluster_col

    for cluster_id in [0, 1, 2]:
        label = map_cluster_labels(cluster_id)
        segment_df = df_with_cluster[df_with_cluster["Cluster"] == cluster_id]

        if segment_df.empty:
            continue

        insights[label] = {
            "customer_count": int(len(segment_df)),
            "churn_rate": float(segment_df["Churned"].mean()),
            "avg_ltv": float(segment_df["Total_LTV"].mean()),
            "avg_age": float(segment_df["Age"].mean()),
            "avg_purchases": float(segment_df["Total_Purchases"].mean()),
            "avg_aov": float(segment_df["Avg_Order_Value"].mean()),
            "avg_support_tickets": float(segment_df["Support_Tickets"].mean()),
        }

    return insights


# ---------------------------------------------------------------------------
# Model Persistence (Save/Load Fitted Models & Scalers)
# ---------------------------------------------------------------------------


def save_models(
    scaler: StandardScaler,
    kmeans: KMeans,
    logistic_model: LogisticRegression,
    ridge_model: Ridge,
    feature_medians: Optional[Dict[str, float]] = None,
    feature_cols: Optional[list] = None,
    filepath: str = "models/pipeline_models.joblib",
) -> None:
    """
    Save all fitted models, scalers and preprocessing metadata to one joblib file.

    Parameters
    ----------
    scaler : StandardScaler
        Fitted feature scaler.
    kmeans : KMeans
        Fitted clustering model.
    logistic_model : LogisticRegression
        Fitted classification model.
    ridge_model : Ridge
        Fitted regression model.
    feature_medians : Dict[str, float], optional
        Training medians used for imputation (train/serve consistency).
    feature_cols : list, optional
        Ordered feature columns the scaler was fitted on.
    filepath : str
        Path where the model bundle will be saved.
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "logistic_model": logistic_model,
        "ridge_model": ridge_model,
        "feature_medians": feature_medians or {},
        "feature_cols": list(feature_cols) if feature_cols else list(MODEL_FEATURES),
        "cluster_feature_cols": list(CLUSTER_FEATURES),
    }

    joblib.dump(bundle, filepath)
    print(f"[PIPELINE] Model bundle saved to {filepath}")


def load_models(
    filepath: str = "models/pipeline_models.joblib",
) -> Dict[str, Any]:
    """
    Load the saved model bundle from disk.

    Parameters
    ----------
    filepath : str
        Path to the joblib model bundle.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing scaler, kmeans, logistic_model, ridge_model.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Model file not found at {filepath}. Train models first."
        )
    bundle = joblib.load(filepath)
    print(f"[PIPELINE] Model bundle loaded from {filepath}")
    return bundle


def list_available_models(
    directory: str = "models",
) -> list:
    """
    List saved model files in a directory.

    Parameters
    ----------
    directory : str
        Directory to search for model files.

    Returns
    -------
    list
        List of file paths for saved model bundles.
    """
    if not os.path.exists(directory):
        return []
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".joblib")
    ]
    return files


# ---------------------------------------------------------------------------
# End-to-End Pipeline Orchestration
# ---------------------------------------------------------------------------


def run_full_pipeline(
    data_path: str = "data/synthetic_customers.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Orchestrate the full ML pipeline from data loading to model training.

    This function runs all stages sequentially and returns a dictionary of
    results, trained models, and metrics.

    Stages:
    1. Load & clean data
    2. Feature scaling
    3. K-Means clustering
    4. Train-test split
    5. Train Logistic Regression (Churn)
    6. Train Ridge Regression (LTV)
    7. Evaluate all models
    8. Save fitted models

    Parameters
    ----------
    data_path : str
        Path to input CSV data.
    test_size : float
        Proportion of test data.
    random_state : int
        Random seed for train-test split.

    Returns
    -------
    Dict[str, Any]
        Comprehensive pipeline results including:
        - models: dict of fitted models
        - scaler: fitted StandardScaler
        - metrics: dict of evaluation metrics
        - predictions: dict of model predictions
        - insights: business segment insights
    """
    print("=" * 70)
    print("FULL ML PIPELINE EXECUTION")
    print("=" * 70)

    # Stage 1: Load data (raw, with injected NaNs in feature columns)
    df_raw = load_data(data_path)

    # Stage 1b: Derive training medians ONCE so both the training-time cleaning
    # and the dashboard's single-record inference share identical imputations.
    feature_medians = compute_feature_medians(df_raw)

    # Stage 2: Clean data (median imputation)
    df = clean_data(df_raw, medians=feature_medians)

    # Stage 3: Select features (full supervised feature set)
    X, feature_cols = select_features(df)

    # Stage 4: Scale features
    X_scaled, scaler = scale_features(X, fit=True)

    # Stage 5: Train K-Means on the behavioural 3-feature subspace only.
    # X_scaled columns follow MODEL_FEATURES order, so slice the cluster
    # columns out of the scaled matrix before feeding K-Means.
    cluster_feature_cols = list(CLUSTER_FEATURES)
    cluster_idx = [feature_cols.index(c) for c in cluster_feature_cols]
    X_cluster = X_scaled[:, cluster_idx]

    kmeans = train_kmeans(X_cluster)
    cluster_labels = assign_clusters(X_cluster, kmeans)
    clustering_metrics = evaluate_clustering(X_cluster, cluster_labels, kmeans)

    # Stage 6: Train-test split (for supervised models).
    # Stratified on the churn label so both classes are present in each fold.
    row_indices = np.arange(len(df))
    (
        X_train,
        X_test,
        y_train_churn,
        y_test_churn,
        y_train_ltv,
        y_test_ltv,
        train_indices,
        test_indices,
    ) = train_test_split(
        X_scaled,
        df[TARGET_COLUMN_CHURN].values,
        df[TARGET_COLUMN_LTV].values,
        row_indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=df[TARGET_COLUMN_CHURN].values,
    )

    print(
        f"[PIPELINE] Train-test split: {len(X_train)} training, "
        f"{len(X_test)} testing samples"
    )

    # Stage 7: Train Logistic Regression (Churn Classification)
    logistic_model = train_logistic_regression(X_train, y_train_churn)

    # Stage 8: Train Ridge Regression (LTV Regression)
    ridge_model = train_ridge_regression(X_train, y_train_ltv)

    # Stage 9: Evaluate models
    churn_metrics = evaluate_classification(logistic_model, X_test, y_test_churn)
    ltv_metrics = evaluate_regression(ridge_model, X_test, y_test_ltv)

    print(f"[PIPELINE] Churn Classification Metrics: {churn_metrics}")
    print(f"[PIPELINE] LTV Regression Metrics: {ltv_metrics}")

    # Stage 10: Save models (bundle includes imputation medians + feature order)
    save_models(
        scaler,
        kmeans,
        logistic_model,
        ridge_model,
        feature_medians=feature_medians,
        feature_cols=feature_cols,
    )

    # Stage 11: Generate predictions on test set
    churn_probs = predict_churn_probability(logistic_model, X_scaled)
    ltv_preds = predict_ltv(ridge_model, X_scaled)

    # Stage 12: Apply decision rules to all customers
    decisions = []
    for i in range(len(df)):
        decision = apply_decision_rules(float(churn_probs[i]), int(cluster_labels[i]))
        decisions.append(decision)

    # Stage 13: Business insights
    insights = get_business_segment_insights(df, cluster_labels)

    # Compile results
    results = {
        "scaler": scaler,
        "kmeans": kmeans,
        "logistic_model": logistic_model,
        "ridge_model": ridge_model,
        "feature_cols": feature_cols,
        "cluster_feature_cols": cluster_feature_cols,
        "cluster_indices": cluster_idx,
        "feature_medians": feature_medians,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "metrics": {
            "churn": churn_metrics,
            "ltv": ltv_metrics,
            "clustering": clustering_metrics,
        },
        "predictions": {
            "churn_probabilities": churn_probs,
            "ltv_predictions": ltv_preds,
            "cluster_labels": cluster_labels,
        },
        "decisions": decisions,
        "insights": insights,
        "data": df,
    }

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    # Entry point: run the full pipeline
    try:
        results = run_full_pipeline()
        print("\nPipeline executed successfully!")
    except FileNotFoundError as e:
        print(f"\nPipeline skipped: {e}")
        print("Run data_generator.py first to create synthetic data.")