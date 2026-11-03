import pickle
import pandas as pd
from pathlib import Path


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FOLDER = PROJECT_ROOT / "ML CP"


# ==========================================
# LOAD MODEL
# ==========================================

model_path = MODEL_FOLDER / "best_model.pkl"

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

    # Convert machine type
    if machine_type == "L":
        type_L = 1
        type_M = 0

    elif machine_type == "M":
        type_L = 0
        type_M = 1

    else:
        type_L = 0
        type_M = 0


    # Create input data
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

    print("\n========== DEBUG INPUT ==========")
    print(input_data)

    print("\n========== MODEL PREDICTION ==========")
    print("Machine Type:", machine_type)
    print("Prediction:", model.predict(input_data)[0])
    print("Probabilities:", model.predict_proba(input_data)[0]) 

    # Predict
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # Determine status
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
        print(
            "Probability:",
            round(result["probability"] * 100, 2),
            "%"
        )
        print("Status:", result["status"]) 

        