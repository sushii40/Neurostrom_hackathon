import pickle
from pathlib import Path
import pandas as pd
import numpy as np


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FOLDER = PROJECT_ROOT / "ML CP"
MODEL_FOLDER.mkdir(exist_ok=True)


# ==========================================
# LOAD OR AUTO-TRAIN MODEL
# ==========================================

model_path = MODEL_FOLDER / "best_model.pkl"

def _train_fallback_model():
    from sklearn.ensemble import RandomForestClassifier
    np.random.seed(42)
    samples = 1000
    data = []
    for i in range(samples):
        air_temp = np.random.uniform(298, 305)
        proc_temp = np.random.uniform(308, 315)
        rpm = np.random.uniform(1400, 1700)
        torque = np.random.uniform(30, 45)
        wear = np.random.uniform(20, 100)
        type_l = 1 if i % 2 == 0 else 0
        type_m = 1 if i % 3 == 0 else 0
        failure = 0
        if i >= samples // 2:
            failure = 1
            fault_type = np.random.choice(["heat", "mech", "wear"])
            if fault_type == "heat":
                air_temp += np.random.uniform(6, 12)
                proc_temp += np.random.uniform(8, 16)
            elif fault_type == "mech":
                torque += np.random.uniform(15, 28)
                rpm += np.random.uniform(-300, 300)
            elif fault_type == "wear":
                wear += np.random.uniform(90, 160)
                torque += np.random.uniform(5, 15)
        data.append([air_temp, proc_temp, rpm, torque, wear, type_l, type_m, failure])

    cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type_L",
        "Type_M",
        "Failure"
    ]
    df = pd.DataFrame(data, columns=cols)
    X = df.drop("Failure", axis=1)
    y = df["Failure"]
    fallback = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    fallback.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(fallback, f)
    return fallback

if not model_path.exists():
    model = _train_fallback_model()
else:
    with open(model_path, "rb") as file:
        model = pickle.load(file)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_machine(
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
    machine_type="L"
):
    if machine_type == "L":
        type_L = 1
        type_M = 0
    elif machine_type == "M":
        type_L = 0
        type_M = 1
    else:
        type_L = 0
        type_M = 0

    input_data = pd.DataFrame([[
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
        type_L,
        type_M
    ]], columns=[
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type_L",
        "Type_M"
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if probability < 0.30:
        status = "NORMAL"
    elif probability < 0.60:
        status = "WARNING"
    else:
        status = "CRITICAL"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "status": status
    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":
    machine_types = ["H", "L", "M"]
    for machine in machine_types:
        result = predict_machine(
            air_temperature=300,
            process_temperature=310,
            rotational_speed=1500,
            torque=40,
            tool_wear=100,
            machine_type=machine
        )
        print("\nMachine:", machine)
        print("Prediction:", result["prediction"])
        print("Probability:", round(result["probability"] * 100, 2), "%")
        print("Status:", result["status"])
