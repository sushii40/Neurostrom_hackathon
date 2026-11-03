import sys
from pathlib import Path

import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT / "simulator"))

from machine_simulator import MachineSimulator


# ==========================================
# GENERATE SENSOR DATA
# ==========================================

def generate_machine_data():

    data = []

    # Normal machines
    normal_machine = MachineSimulator(
        machine_type="L",
        fault_mode="NORMAL"
    )

    for _ in range(150):

        sensor = normal_machine.generate_sensor_data()

        data.append({
            "air_temperature": sensor["air_temperature"],
            "process_temperature": sensor["process_temperature"],
            "rotational_speed": sensor["rotational_speed"],
            "torque": sensor["torque"],
            "tool_wear": sensor["tool_wear"],
            "failure": 0
        })

    # Faulty machines
    fault_machine = MachineSimulator(
        machine_type="L",
        fault_mode="MECHANICAL_FAULT"
    )

    for _ in range(150):

        sensor = fault_machine.generate_sensor_data()

        data.append({
            "air_temperature": sensor["air_temperature"],
            "process_temperature": sensor["process_temperature"],
            "rotational_speed": sensor["rotational_speed"],
            "torque": sensor["torque"],
            "tool_wear": sensor["tool_wear"],
            "failure": 1
        })

    return pd.DataFrame(data)


# ==========================================
# CONVERT SENSOR VALUES INTO CONDITIONS
# ==========================================

def create_transactions(df):

    transactions = pd.DataFrame()

    transactions["High_Air_Temperature"] = (
        df["air_temperature"] > df["air_temperature"].median()
    )

    transactions["High_Process_Temperature"] = (
        df["process_temperature"] > df["process_temperature"].median()
    )

    transactions["High_Torque"] = (
        df["torque"] > 50
    )

    transactions["High_Tool_Wear"] = (
        df["tool_wear"] > 100
    )

    transactions["Low_Rotational_Speed"] = (
        df["rotational_speed"] < 1400
    )

    transactions["Failure"] = (
        df["failure"] == 1
    )

    return transactions.astype(bool)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("\n========================================")
    print("        ASSOCIATION RULE MINING")
    print("              APRIORI")
    print("========================================")

    # Generate data
    df = generate_machine_data()

    print("\nMachine data generated:")
    print("Total samples:", len(df))

    # Create transactions
    transactions = create_transactions(df)

    print("\nTransaction conditions created.")

    # ======================================
    # APRIORI
    # ======================================

    frequent_itemsets = apriori(
        transactions,
        min_support=0.10,
        use_colnames=True
    )

    print("\n========================================")
    print("         FREQUENT ITEMSETS")
    print("========================================")

    if frequent_itemsets.empty:

        print("No frequent itemsets found.")

    else:

        frequent_itemsets = frequent_itemsets.sort_values(
            by="support",
            ascending=False
        )

        print(
            frequent_itemsets.head(15).to_string(
                index=False
            )
        )

    # ======================================
    # ASSOCIATION RULES
    # ======================================

    print("\n========================================")
    print("          ASSOCIATION RULES")
    print("========================================")

    if frequent_itemsets.empty:

        print("No rules can be generated.")

    else:

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=0.60
        )

        if rules.empty:

            print("No strong association rules found.")

        else:

            rules = rules.sort_values(
                by="lift",
                ascending=False
            )

            display_columns = [
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift"
            ]

            print(
                rules[display_columns]
                .head(15)
                .to_string(index=False)
            )

            # ==================================
            # FAILURE-RELATED RULES
            # ==================================

            failure_rules = rules[
                rules["consequents"].apply(
                    lambda x: "Failure" in x
                )
            ]

            print("\n========================================")
            print("       FAILURE-RELATED RULES")
            print("========================================")

            if failure_rules.empty:

                print("No failure-related rules found.")

            else:

                failure_rules = failure_rules.sort_values(
                    by="lift",
                    ascending=False
                )

                print(
                    failure_rules[display_columns]
                    .head(10)
                    .to_string(index=False)
                )

                # Save results
                output_file = (
                    PROJECT_ROOT
                    / "association_rules"
                    / "association_rules_results.csv"
                )

                failure_rules[display_columns].to_csv(
                    output_file,
                    index=False
                )

                print(
                    "\nRules saved to:",
                    output_file
                )

    print("\n========================================")
    print(" Association Rule Mining Completed!")
    print("========================================")