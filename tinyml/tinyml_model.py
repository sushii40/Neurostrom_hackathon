# ============================================================
# TINYML PREDICTIVE MAINTENANCE
# Adaptive Multimodal Predictive Maintenance Project
# ============================================================

import os
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TINYML_FOLDER = os.path.join(PROJECT_ROOT, "tinyml")

os.makedirs(TINYML_FOLDER, exist_ok=True)


# ============================================================
# 2. GENERATE LIGHTWEIGHT PREDICTIVE MAINTENANCE DATA
# ============================================================

print("=" * 60)
print("        TINYML PREDICTIVE MAINTENANCE")
print("=" * 60)

print("\nGenerating lightweight dataset...")

np.random.seed(42)

samples = 1000

data = []

for i in range(samples):

    # Normal machine values
    air_temperature = np.random.uniform(298, 305)
    process_temperature = np.random.uniform(308, 315)
    rotational_speed = np.random.uniform(1400, 1700)
    torque = np.random.uniform(30, 45)
    tool_wear = np.random.uniform(20, 100)

    # Generate failure cases
    if i >= samples // 2:

        failure_type = np.random.choice([
            "OVERHEATING",
            "MECHANICAL_FAULT",
            "TOOL_WEAR"
        ])

        if failure_type == "OVERHEATING":
            air_temperature += np.random.uniform(5, 12)
            process_temperature += np.random.uniform(8, 15)

        elif failure_type == "MECHANICAL_FAULT":
            torque += np.random.uniform(15, 25)
            rotational_speed += np.random.uniform(-300, 300)

        elif failure_type == "TOOL_WEAR":
            tool_wear += np.random.uniform(80, 150)
            torque += np.random.uniform(5, 12)

        failure = 1

    else:
        failure = 0

    data.append([
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
        failure
    ])


columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Failure"
]

df = pd.DataFrame(data, columns=columns)

print("Dataset generated successfully.")
print("Total samples:", len(df))

print("\nClass distribution:")
print(df["Failure"].value_counts())


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

X = df.drop("Failure", axis=1)
y = df["Failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. TRAIN LIGHTWEIGHT MODEL
# ============================================================

print("\nTraining TinyML model...")

start_training = time.time()

# Small decision tree suitable for edge deployment
tinyml_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    random_state=42
)

tinyml_model.fit(X_train_scaled, y_train)

training_time = time.time() - start_training

print("TinyML model trained successfully.")


# ============================================================
# 6. MODEL PREDICTION
# ============================================================

start_inference = time.perf_counter()

predictions = tinyml_model.predict(X_test_scaled)

end_inference = time.perf_counter()

total_inference_time = end_inference - start_inference

average_inference_time = (
    total_inference_time / len(X_test)
) * 1000


# ============================================================
# 7. MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(y_test, predictions)

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


print("\n" + "=" * 60)
print("             TINYML MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")

print(f"\nTraining Time  : {training_time:.6f} seconds")
print(
    f"Avg Inference  : "
    f"{average_inference_time:.6f} ms/sample"
)


# ============================================================
# 8. MODEL COMPLEXITY
# ============================================================

print("\n" + "=" * 60)
print("             MODEL COMPLEXITY")
print("=" * 60)

print("Tree Depth     :", tinyml_model.get_depth())
print("Tree Leaves    :", tinyml_model.get_n_leaves())


# ============================================================
# 9. SAVE TINYML MODEL
# ============================================================

model_path = os.path.join(
    TINYML_FOLDER,
    "tinyml_model.pkl"
)

scaler_path = os.path.join(
    TINYML_FOLDER,
    "tinyml_scaler.pkl"
)

with open(model_path, "wb") as file:
    pickle.dump(tinyml_model, file)

with open(scaler_path, "wb") as file:
    pickle.dump(scaler, file)


# ============================================================
# 10. CHECK MODEL SIZE
# ============================================================

model_size = os.path.getsize(model_path)
scaler_size = os.path.getsize(scaler_path)

total_size = model_size + scaler_size

print("\n" + "=" * 60)
print("             MODEL SIZE")
print("=" * 60)

print(f"Model Size     : {model_size / 1024:.2f} KB")
print(f"Scaler Size    : {scaler_size / 1024:.2f} KB")
print(f"Total Size     : {total_size / 1024:.2f} KB")


# ============================================================
# 11. TEST SINGLE MACHINE READING
# ============================================================

print("\n" + "=" * 60)
print("          SINGLE MACHINE PREDICTION")
print("=" * 60)

sample_machine = np.array([[
    310.0,     # Air temperature
    320.0,     # Process temperature
    1250.0,    # Rotational speed
    62.0,      # Torque
    180.0      # Tool wear
]])

sample_scaled = scaler.transform(sample_machine)

single_prediction = tinyml_model.predict(sample_scaled)[0]

single_probability = tinyml_model.predict_proba(
    sample_scaled
)[0][1]


if single_prediction == 1:
    machine_status = "FAILURE RISK"
else:
    machine_status = "NORMAL"


print("Air Temperature    :", sample_machine[0][0])
print("Process Temperature:", sample_machine[0][1])
print("Rotational Speed   :", sample_machine[0][2])
print("Torque             :", sample_machine[0][3])
print("Tool Wear          :", sample_machine[0][4])

print("\nPrediction          :", int(single_prediction))
print(
    "Failure Probability : "
    f"{single_probability * 100:.2f}%"
)
print("Machine Status      :", machine_status)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Training Time (seconds)",
        "Average Inference Time (ms)",
        "Tree Depth",
        "Tree Leaves",
        "Model Size (KB)",
        "Scaler Size (KB)",
        "Total Model Size (KB)"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        training_time,
        average_inference_time,
        tinyml_model.get_depth(),
        tinyml_model.get_n_leaves(),
        model_size / 1024,
        scaler_size / 1024,
        total_size / 1024
    ]
})

results_path = os.path.join(
    TINYML_FOLDER,
    "tinyml_results.csv"
)

results.to_csv(results_path, index=False)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("             TINYML COMPLETED")
print("=" * 60)

print("\nFiles saved:")

print(model_path)
print(scaler_path)
print(results_path)

print("\nTinyML model is ready for edge-device deployment.")
