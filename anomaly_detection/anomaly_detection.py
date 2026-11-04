import os
import sys
from pathlib import Path

# Silence core-counting warnings on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import homogeneity_score, completeness_score
from sklearn.cluster import AgglomerativeClustering

# Allow importing simulator
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "simulator"))

from machine_simulator import MachineSimulator


# ==========================================
# GENERATE MACHINE SENSOR DATA
# ==========================================

def generate_data():

    data = []

    # --------------------------------------
    # NORMAL DATA
    # --------------------------------------

    machine = MachineSimulator(
        machine_type="L",
        fault_mode="NORMAL"
    )

    for _ in range(150):

        sensor = machine.generate_sensor_data()

        data.append({
            "air_temperature": sensor["air_temperature"],
            "process_temperature": sensor["process_temperature"],
            "rotational_speed": sensor["rotational_speed"],
            "torque": sensor["torque"],
            "tool_wear": sensor["tool_wear"],
            "true_label": 0
        })

    # --------------------------------------
    # FAULT DATA
    # --------------------------------------

    machine = MachineSimulator(
        machine_type="L",
        fault_mode="MECHANICAL_FAULT"
    )

    for _ in range(150):

        sensor = machine.generate_sensor_data()

        data.append({
            "air_temperature": sensor["air_temperature"],
            "process_temperature": sensor["process_temperature"],
            "rotational_speed": sensor["rotational_speed"],
            "torque": sensor["torque"],
            "tool_wear": sensor["tool_wear"],
            "true_label": 1
        })

    return pd.DataFrame(data)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("\n========================================")
    print("       MACHINE ANOMALY DETECTION")
    print("========================================")

    # Generate data
    df = generate_data()

    print("\nDataset Generated")
    print("----------------------------------------")
    print("Total samples :", len(df))
    print("Normal samples:", sum(df["true_label"] == 0))
    print("Fault samples :", sum(df["true_label"] == 1))

    # ======================================
    # FEATURES
    # ======================================

    features = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear"
    ]

    X = df[features]
    y_true = df["true_label"]

    # ======================================
    # STANDARDIZATION
    # ======================================

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ======================================
    # K-MEANS
    # ======================================

    print("\n\n========== K-MEANS ==========")

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(X_scaled)

    kmeans_homogeneity = homogeneity_score(
        y_true,
        kmeans_labels
    )

    kmeans_completeness = completeness_score(
        y_true,
        kmeans_labels
    )

    print("Homogeneity :", round(kmeans_homogeneity, 4))
    print("Completeness:", round(kmeans_completeness, 4))

    # ======================================
    # DBSCAN
    # ======================================

    print("\n\n========== DBSCAN ==========")

    dbscan = DBSCAN(
        eps=0.8,
        min_samples=5
    )

    dbscan_labels = dbscan.fit_predict(X_scaled)

    # DBSCAN uses -1 for anomalies
    dbscan_anomalies = np.sum(dbscan_labels == -1)

    print("Clusters detected :", len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0))
    print("Anomalies detected:", dbscan_anomalies)

    # ======================================
    # HIERARCHICAL CLUSTERING
    # ======================================

    print("\n\n========== HIERARCHICAL CLUSTERING ==========")

    hierarchical = AgglomerativeClustering(
        n_clusters=2
    )

    hierarchical_labels = hierarchical.fit_predict(X_scaled)

    hierarchical_homogeneity = homogeneity_score(
        y_true,
        hierarchical_labels
    )

    hierarchical_completeness = completeness_score(
        y_true,
        hierarchical_labels
    )

    print("Homogeneity :", round(hierarchical_homogeneity, 4))
    print("Completeness:", round(hierarchical_completeness, 4))

    # ======================================
    # SUMMARY
    # ======================================

    print("\n\n========================================")
    print("           ANOMALY DETECTION")
    print("                SUMMARY")
    print("========================================")

    print(
        f"K-Means       -> "
        f"Homogeneity={kmeans_homogeneity:.4f}, "
        f"Completeness={kmeans_completeness:.4f}"
    )

    print(
        f"DBSCAN        -> "
        f"Anomalies={dbscan_anomalies}"
    )

    print(
        f"Hierarchical  -> "
        f"Homogeneity={hierarchical_homogeneity:.4f}, "
        f"Completeness={hierarchical_completeness:.4f}"
    )

    print("\nAnomaly detection completed successfully!")
