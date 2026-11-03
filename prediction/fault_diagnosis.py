# ============================================================
# INTELLIGENT FAULT DIAGNOSIS MODULE
# ============================================================

def diagnose_fault(
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
):

    possible_faults = []
    recommendations = []

    temperature_difference = (
        process_temperature - air_temperature
    )

    # ========================================================
    # 1. OVERHEATING DETECTION
    # ========================================================

    if (
        air_temperature > 310
        or process_temperature > 320
        or temperature_difference > 15
    ):

        possible_faults.append("POSSIBLE OVERHEATING")

        recommendations.append(
            "Inspect cooling system and temperature sensors."
        )

    # ========================================================
    # 2. MECHANICAL FAULT DETECTION
    # ========================================================

    if torque > 55 or rotational_speed < 1200 or rotational_speed > 1800:

        possible_faults.append("POSSIBLE MECHANICAL FAULT")

        recommendations.append(
            "Inspect motor, bearings and rotating components."
        )

    # ========================================================
    # 3. TOOL WEAR DETECTION
    # ========================================================

    if tool_wear > 200:

        possible_faults.append("EXCESSIVE TOOL WEAR")

        recommendations.append(
            "Inspect and consider replacing the worn tool."
        )

    # ========================================================
    # FINAL DIAGNOSIS
    # ========================================================

    if not possible_faults:

        diagnosis = "NO OBVIOUS FAULT DETECTED"

        recommendations.append(
            "Continue normal monitoring."
        )

    else:

        diagnosis = ", ".join(possible_faults)

    return {
        "diagnosis": diagnosis,
        "possible_faults": possible_faults,
        "recommendations": recommendations,
        "temperature_difference": temperature_difference
    }

if __name__ == "__main__":

    result = diagnose_fault(
        air_temperature=315,
        process_temperature=325,
        rotational_speed=1500,
        torque=40,
        tool_wear=100
    )

    print("Diagnosis:", result["diagnosis"])

    print("\nPossible Faults:")

    for fault in result["possible_faults"]:
        print("-", fault)

    print("\nRecommendations:")

    for recommendation in result["recommendations"]:
        print("-", recommendation)


        