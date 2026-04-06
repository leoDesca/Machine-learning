"""
train.py  —  Group 17 Supply Chain: HDBSCAN Training (Deployment-Aligned)
===========================================================================
Trains HDBSCAN using the exact 6 fields exposed by the deployment form.

Run:
    python train.py

Outputs written to model/:
    hdbscan_model.pkl
    scaler.pkl
    cluster_info.json
    feature_names.json
    metrics.json
"""

import json
import os

import hdbscan
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_FILE = "makerere_Cafeteria_synthetic.csv"
FEATURE_NAMES = [
    "Daily_Prepared",
    "Daily_Sold",
    "Daily_Revenue",
    "Daily_Profit",
    "DayOfWeekNum",
    "Month",
]


def _metric_or_none(name: str, X: np.ndarray, labels: np.ndarray):
    non_noise = labels != -1
    unique_non_noise = set(labels[non_noise])
    if len(unique_non_noise) < 2:
        return None

    if name == "silhouette":
        return float(silhouette_score(X[non_noise], labels[non_noise]))
    if name == "davies_bouldin":
        return float(davies_bouldin_score(X[non_noise], labels[non_noise]))
    if name == "calinski_harabasz":
        return float(calinski_harabasz_score(X[non_noise], labels[non_noise]))
    return None


def _build_cluster_info(df: pd.DataFrame, labels: np.ndarray) -> dict:
    data = df.copy()
    data["Cluster"] = labels

    cluster_info = {}
    non_noise_clusters = sorted([c for c in set(labels) if c != -1])

    # Rank non-noise clusters by average sold volume to assign business labels.
    sold_rank = []
    for c in non_noise_clusters:
        sub = data[data["Cluster"] == c]
        sold_rank.append((c, float(sub["Daily_Sold"].mean())))
    sold_rank.sort(key=lambda x: x[1], reverse=True)

    high_volume_cluster = sold_rank[0][0] if sold_rank else None

    for c in sorted(set(labels)):
        sub = data[data["Cluster"] == c]
        key = str(c)

        if c == -1:
            cluster_info[key] = {
                "name": "Anomalous / Atypical Day",
                "supply_alert": "REVIEW",
                "avg_daily_sold": 0,
                "supply_action": (
                    "This day does not match any learned operational pattern. "
                    "Flag for manual procurement review and use conservative stock."
                ),
                "size": int(len(sub)),
            }
            continue

        avg_sold = float(sub["Daily_Sold"].mean())
        avg_revenue = float(sub["Daily_Revenue"].mean())
        avg_profit = float(sub["Daily_Profit"].mean())

        if c == high_volume_cluster and len(non_noise_clusters) > 1:
            name = "High-Volume Operational Days"
            alert = "CRITICAL"
            action = (
                "Demand is in a high-volume regime. Increase procurement, order earlier, "
                "and monitor sell-through every 2 hours."
            )
        else:
            name = "Standard Operational Days"
            alert = "NORMAL"
            action = (
                "Use normal procurement anchored to recent averages and adjust for day-specific variation."
            )

        cluster_info[key] = {
            "name": name,
            "supply_alert": alert,
            "avg_daily_sold": round(avg_sold, 1),
            "avg_daily_revenue": round(avg_revenue, 1),
            "avg_daily_profit": round(avg_profit, 1),
            "supply_action": action,
            "size": int(len(sub)),
        }

    return cluster_info


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"'{DATA_FILE}' not found.")

    print(f"Loading {DATA_FILE}...")
    cafe = pd.read_csv(DATA_FILE)
    cafe["Date"] = pd.to_datetime(cafe["Date"])

    for col in ["Revenue_UGX", "Gross_Profit_UGX"]:
        cafe[col] = pd.to_numeric(cafe[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

    daily = cafe.groupby(["Date", "Cafeteria_ID"], observed=True).agg(
        Daily_Prepared=("Portions_Prepared", "sum"),
        Daily_Sold=("Portions_Sold", "sum"),
        Daily_Revenue=("Revenue_UGX", "sum"),
        Daily_Profit=("Gross_Profit_UGX", "sum"),
    ).reset_index()

    daily["DayOfWeekNum"] = daily["Date"].dt.dayofweek
    daily["Month"] = daily["Date"].dt.month

    X_df = daily[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).dropna()
    X_raw = X_df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print("Fitting HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=20,
        min_samples=20,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int((labels == -1).sum())
    noise_pct = noise_count / len(labels) * 100

    sil = _metric_or_none("silhouette", X_scaled, labels)
    dbi = _metric_or_none("davies_bouldin", X_scaled, labels)
    chi = _metric_or_none("calinski_harabasz", X_scaled, labels)

    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {noise_count} ({noise_pct:.2f}%)")

    cluster_info = _build_cluster_info(X_df, labels)

    os.makedirs("model", exist_ok=True)
    joblib.dump(clusterer, "model/hdbscan_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    with open("model/cluster_info.json", "w", encoding="utf-8") as f:
        json.dump(cluster_info, f, indent=2)

    with open("model/feature_names.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    with open("model/metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "HDBSCAN (deployment-aligned 6-feature training)",
                "min_cluster_size": 20,
                "min_samples": 20,
                "prediction_data": True,
                "n_clusters": n_clusters,
                "noise_count": noise_count,
                "noise_pct": round(noise_pct, 2),
                "silhouette": round(sil, 4) if sil is not None else None,
                "davies_bouldin": round(dbi, 4) if dbi is not None else None,
                "calinski_harabasz": round(chi, 1) if chi is not None else None,
                "n_samples": int(X_raw.shape[0]),
                "n_features": int(X_raw.shape[1]),
                "feature_names": FEATURE_NAMES,
            },
            f,
            indent=2,
        )

    print("Saved model artifacts to model/.")


if __name__ == "__main__":
    main()
