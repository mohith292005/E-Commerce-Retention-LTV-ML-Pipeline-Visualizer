"""
data_generator.py

Generates synthetic customer data for E-Commerce Customer Retention & LTV Prediction.

Dataset features:
- Customer_ID: Unique identifier (int)
- Age: Customer age in years (int, 18-70)
- Days_Since_Last_Purchase: Days since last purchase (int, 1-365)
- Total_Purchases: Total number of purchases (int, 1-50)
- Avg_Order_Value: Average order value in dollars (float, 20.0-500.0)
- Support_Tickets: Number of support tickets filed (int, 0-10)
- Churned: Whether customer has churned (0 = Active, 1 = Churned) (int, binary)
- Total_LTV: Total Lifetime Value in dollars (float, 100.0-10000.0)

Design notes:
- The churn label is drawn from a weighted behavioural risk score and split at
  the 75th percentile, so exactly ~25% of customers churn. This keeps the
  classification target non-degenerate (ROC-AUC needs both classes present).
- ~5% missing values are injected ONLY into the five *feature* columns so the
  downstream pipeline can demonstrate median imputation. Customer_ID and the
  supervision targets (Churned / Total_LTV) are kept intact because duplicated
  IDs or NaN labels would corrupt evaluation.
"""

import pandas as pd
import numpy as np
import random
import os


def generate_synthetic_customer_data(
    n_customers: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic dataset of customers for E-Commerce Retention Analysis.

    Parameters
    ----------
    n_customers : int
        Number of customers to generate (default: 1000).
    random_seed : int
        Seed for reproducibility (default: 42).

    Returns
    -------
    pd.DataFrame
        DataFrame containing synthetic customer data with the following columns:
        - Customer_ID
        - Age
        - Days_Since_Last_Purchase
        - Total_Purchases
        - Avg_Order_Value
        - Support_Tickets
        - Churned
        - Total_LTV
    """
    # Seed both RNGs for full reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)

    # --- Customer ID ---
    customer_ids = np.arange(1, n_customers + 1)

    # --- Age ---
    # Normally distributed around 45, bounded [18, 70]
    ages = np.random.normal(loc=45, scale=12, size=n_customers)
    ages = np.clip(ages, 18, 70).astype(int)

    # --- Days Since Last Purchase ---
    # Exponential decay: most recent purchases, some dormant
    days_since_last = np.random.exponential(scale=30, size=n_customers)
    days_since_last = np.clip(days_since_last, 1, 365).astype(int)

    # --- Total Purchases ---
    # Poisson-like distribution: most customers have few purchases
    total_purchases = np.random.poisson(lam=8, size=n_customers)
    total_purchases = np.clip(total_purchases, 1, 50).astype(int)

    # --- Average Order Value ---
    # Correlated with total purchases and customer age/wealth
    base_aov = np.random.normal(loc=80, scale=20, size=n_customers)
    # Adjust based on total purchases (more purchases -> higher or varied AOV)
    aov_adjustment = (total_purchases / 8.0) * 10
    avg_order_values = np.clip(base_aov + aov_adjustment, 20, 500).round(2)

    # --- Support Tickets ---
    # Slight correlation: customers with many purchases or high days since last purchase
    # may have more support issues
    support_base = np.random.poisson(lam=1, size=n_customers)
    # Increase tickets for customers who are "at risk" (high days since last purchase)
    support_boost = (days_since_last > 180).astype(int) * 2
    support_tickets = np.clip(support_base + support_boost, 0, 10).astype(int)

    # --- Churn Label ---
    # Continuous risk score blending the drivers of churn:
    #   - long dormancy (high Days_Since_Last_Purchase)
    #   - few lifetime purchases
    #   - frequent support issues
    #   - young / elderly age brackets (less loyal segments)
    #   - low average order value (bargain hunters)
    churn_score = (
        1.6 * (days_since_last / 180.0)
        + 1.1 * np.clip(1.0 - total_purchases / 12.0, 0.0, 1.0)
        + 0.8 * (support_base + support_boost) / 6.0
        + 0.4 * ((ages < 30) | (ages > 60)).astype(float)
        + 0.5 * np.clip(1.0 - avg_order_values / 150.0, 0.0, 1.0)
        + np.random.normal(0.0, 0.35, n_customers)  # stochastic noise
    )
    # Threshold at a quantile so the dataset has a realistic ~25% churn rate
    # (guarantees a balanced-enough target for ROC-AUC / confusion matrix).
    target_churn_rate = 0.25
    churn_threshold = np.quantile(churn_score, 1.0 - target_churn_rate)
    churned = (churn_score >= churn_threshold).astype(int)

    # --- Total LTV ---
    # LTV correlates with total purchases and AOV; churned customers stop
    # generating revenue, so their realized LTV is discounted.
    base_ltv = total_purchases * avg_order_values
    churn_discount = np.where(churned == 1, 0.65, 1.0)
    ltv_noise = np.random.normal(1.0, 0.12, n_customers)
    total_ltv = np.clip(
        base_ltv * churn_discount * ltv_noise, 100, 10000
    ).round(2)

    # Build DataFrame
    df = pd.DataFrame(
        {
            "Customer_ID": customer_ids,
            "Age": ages,
            "Days_Since_Last_Purchase": days_since_last,
            "Total_Purchases": total_purchases,
            "Avg_Order_Value": avg_order_values,
            "Support_Tickets": support_tickets,
            "Churned": churned,
            "Total_LTV": total_ltv,
        }
    )

    # --- Introduce some missing values randomly (feature columns only) ---
    # ~5% of feature cells become NaN to exercise the median-imputation stage.
    # Targets (Churned, Total_LTV) and the ID column are kept intact.
    feature_columns = [
        "Age",
        "Days_Since_Last_Purchase",
        "Total_Purchases",
        "Avg_Order_Value",
        "Support_Tickets",
    ]
    for col in feature_columns:
        mask = np.random.random(n_customers) < 0.05
        df.loc[mask, col] = np.nan

    # Shuffle the index to make it more realistic
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return df


def save_data_to_csv(
    df: pd.DataFrame,
    filepath: str = "data/synthetic_customers.csv",
) -> None:
    """
    Save the synthetic DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    filepath : str
        Path where the CSV will be saved. Directory will be created if it doesn't exist.
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    df.to_csv(filepath, index=False)
    print(f"[DATA] Synthetic data saved to {filepath} ({len(df)} customers)")


def load_data_from_csv(filepath: str = "data/synthetic_customers.csv") -> pd.DataFrame:
    """
    Load synthetic customer data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}. Generate data first.")
    df = pd.read_csv(filepath)
    print(f"[DATA] Loaded {len(df)} customers from {filepath}")
    return df


def main():
    """Entry point: generate and save synthetic data."""
    print("=" * 60)
    print("SYNTHETIC CUSTOMER DATA GENERATOR")
    print("=" * 60)

    # Generate data
    df = generate_synthetic_customer_data(n_customers=1000, random_seed=42)

    # Display brief summary
    print("\nDataset Overview:")
    print(df.head())
    print("\nClass distribution (Churned):")
    print(df["Churned"].value_counts())
    print(f"\nBasic stats:\n{df.describe().T}")

    # Save to CSV
    save_data_to_csv(df, filepath="data/synthetic_customers.csv")

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()