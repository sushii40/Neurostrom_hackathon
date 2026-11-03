import sys
from pathlib import Path
import joblib  

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT / "simulator"))

from machine_simulator import MachineSimulator


# ==========================================
# GENERATE MULTIMODAL DATA
# ==========================================

def generate_multimodal_data():

    data = []

    fault_modes = [
        ("NORMAL", 0),
        ("MECHANICAL_FAULT", 1),
        ("OVERHEATING", 1),
        ("TOOL_WEAR", 1)
    ]

    samples_per_mode = 100

    for fault_mode, label in fault_modes:

        machine = MachineSimulator(
            machine_type="L",
            fault_mode=fault_mode
        )

        for _ in range(samples_per_mode):

            sensor = machine.generate_sensor_data()

            # ----------------------------------
            # SENSOR FEATURES
            # ----------------------------------

            air_temp = sensor["air_temperature"]
            process_temp = sensor["process_temperature"]
            rpm = sensor["rotational_speed"]
            torque = sensor["torque"]
            tool_wear = sensor["tool_wear"]

            # ----------------------------------
            # SIMULATED VIBRATION / AUDIO
            # ----------------------------------

            # Vibration increases with mechanical stress
            vibration_rms = (
                0.5
                + abs(torque - 40) * 0.035
                + np.random.normal(0, 0.05)
            )

            # Frequency-related feature
            vibration_frequency = (
                rpm / 60
                + np.random.normal(0, 2)
            )

            # Audio energy
            audio_energy = (
                0.2
                + abs(torque - 40) * 0.02
                + np.random.normal(0, 0.02)
            )

            # Zero-crossing-like feature
            zero_crossing_rate = (
                0.05
                + abs(rpm - 1500) / 15000
                + np.random.normal(0, 0.005)
            )

            # Faults create additional vibration/noise
            if fault_mode != "NORMAL":

                vibration_rms += np.random.uniform(
                    0.2, 0.5
                )

                audio_energy += np.random.uniform(
                    0.05, 0.15
                )

            data.append({

                # Sensor modality
                "air_temperature": air_temp,
                "process_temperature": process_temp,
                "rotational_speed": rpm,
                "torque": torque,
                "tool_wear": tool_wear,

                # Simulated audio/vibration modality
                "vibration_rms": vibration_rms,
                "vibration_frequency": vibration_frequency,
                "audio_energy": audio_energy,
                "zero_crossing_rate": zero_crossing_rate,

                # Ground truth
                "failure": label
            })

    return pd.DataFrame(data)


# ==========================================
# EVALUATE MODEL
# ==========================================

def evaluate_model(name, X_train, X_test, y_train, y_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(f"\n{name}")
    print("----------------------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("\n========================================")
    print("       MULTIMODAL FUSION")
    print("    SENSOR + AUDIO/VIBRATION")
    print("========================================")

    # Generate data
    df = generate_multimodal_data()

    print("\nDataset generated successfully.")
    print("Total samples:", len(df))

    print("\nClass distribution:")
    print(df["failure"].value_counts())

    # ======================================
    # SENSOR FEATURES ONLY
    # ======================================

    sensor_features = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear"
    ]

    # ======================================
    # AUDIO/VIBRATION FEATURES ONLY
    # ======================================

    audio_features = [
        "vibration_rms",
        "vibration_frequency",
        "audio_energy",
        "zero_crossing_rate"
    ]

    # ======================================
    # COMBINED FEATURES
    # ======================================

    multimodal_features = (
        sensor_features
        + audio_features
    )

    y = df["failure"]

    # Same split for fair comparison
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # ======================================
    # SENSOR MODEL
    # ======================================

    sensor_results = evaluate_model(
        "SENSOR-ONLY MODEL",
        df[sensor_features].iloc[train_idx],
        df[sensor_features].iloc[test_idx],
        y_train,
        y_test
    )

    # ======================================
    # AUDIO/VIBRATION MODEL
    # ======================================

    audio_results = evaluate_model(
        "AUDIO/VIBRATION-ONLY MODEL",
        df[audio_features].iloc[train_idx],
        df[audio_features].iloc[test_idx],
        y_train,
        y_test
    )

    # ======================================
    # MULTIMODAL MODEL
    # ======================================

    multimodal_results = evaluate_model(
        "MULTIMODAL MODEL",
        df[multimodal_features].iloc[train_idx],
        df[multimodal_features].iloc[test_idx],
        y_train,
        y_test
    )
        # ======================================
    # TRAIN FINAL MODEL AND SAVE ARTIFACTS
    # ======================================

    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(
        df[multimodal_features]
    )

    final_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    final_model.fit(X_full_scaled, y)

    model_dir = PROJECT_ROOT / "multimodal"

    joblib.dump(
        final_model,
        model_dir / "multimodal_model.pkl"
    )

    joblib.dump(
        final_scaler,
        model_dir / "multimodal_scaler.pkl"
    )

    joblib.dump(
        multimodal_features,
        model_dir / "multimodal_features.pkl"
    )

    print("\nMultimodal model, scaler and feature list saved.")
    
    # ======================================
    # COMPARISON
    # ======================================

    results = pd.DataFrame([
        sensor_results,
        audio_results,
        multimodal_results
    ])

    print("\n========================================")
    print("          MODEL COMPARISON")
    print("========================================")

    print(
        results.to_string(index=False)
    )

    # ======================================
    # SAVE RESULTS
    # ======================================

    output_file = (
        PROJECT_ROOT
        / "multimodal"
        / "multimodal_results.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    # Save generated multimodal dataset
    dataset_file = (
        PROJECT_ROOT
        / "multimodal"
        / "multimodal_dataset.csv"
    )

    df.to_csv(
        dataset_file,
        index=False
    )

    print("\nResults saved to:")
    print(output_file)

    print("\nDataset saved to:")
    print(dataset_file)

    print("\n========================================")
    print("       MULTIMODAL FUSION COMPLETED")
    print("========================================")
    