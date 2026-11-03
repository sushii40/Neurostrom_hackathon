import sys
from pathlib import Path
import random
import pickle

# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT / "prediction"))
sys.path.append(str(PROJECT_ROOT / "simulator"))

from prediction_pipeline import predict_machine
from machine_simulator import MachineSimulator


# ==========================================
# Q-LEARNING PARAMETERS
# ==========================================

ACTIONS = [
    "CONTINUE_OPERATION",
    "MONITOR",
    "SCHEDULE_MAINTENANCE",
    "EMERGENCY_SHUTDOWN"
]

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

EPISODES = 500


# ==========================================
# STATE CONVERSION
# ==========================================

def get_state(probability, torque, tool_wear):

    if probability < 0.30:
        risk = 0
    elif probability < 0.60:
        risk = 1
    else:
        risk = 2

    if torque < 50:
        torque_level = 0
    elif torque < 65:
        torque_level = 1
    else:
        torque_level = 2

    if tool_wear < 100:
        wear_level = 0
    elif tool_wear < 180:
        wear_level = 1
    else:
        wear_level = 2

    return (risk, torque_level, wear_level)


# ==========================================
# REWARD FUNCTION
# ==========================================

def get_reward(state, action):

    risk, torque_level, wear_level = state

    if risk == 0:

        if action == 0:
            return 10
        elif action == 1:
            return 3
        elif action == 2:
            return -5
        elif action == 3:
            return -10

    elif risk == 1:

        if action == 0:
            return -5
        elif action == 1:
            return 8
        elif action == 2:
            return 10
        elif action == 3:
            return 2

    else:

        if action == 0:
            return -20
        elif action == 1:
            return -10
        elif action == 2:
            return 8
        elif action == 3:
            return 15

    return 0


# ==========================================
# CHOOSE ACTION
# ==========================================

def choose_action(q_table, state, epsilon):

    if random.random() < epsilon:

        return random.randint(0, len(ACTIONS) - 1)

    return max(
        range(len(ACTIONS)),
        key=lambda action: q_table.get(
            state,
            [0.0] * len(ACTIONS)
        )[action]
    )


# ==========================================
# TRAIN Q-LEARNING AGENT
# ==========================================

def train_agent():

    q_table = {}

    epsilon = EPSILON

    total_rewards = []

    print("\n========================================")
    print("     ADAPTIVE Q-LEARNING TRAINING")
    print("========================================")

    fault_modes = [
        "NORMAL",
        "OVERHEATING",
        "MECHANICAL_FAULT",
        "TOOL_WEAR"
    ]

    for episode in range(EPISODES):

        fault_mode = random.choice(fault_modes)

        machine = MachineSimulator(
            machine_type="L",
            fault_mode=fault_mode
        )

        episode_reward = 0

        # Initial sensor reading
        sensor = machine.generate_sensor_data()

        for step in range(10):

            if not machine.machine_running:
                break

            # 1. ML prediction
            result = predict_machine(
                sensor["air_temperature"],
                sensor["process_temperature"],
                sensor["rotational_speed"],
                sensor["torque"],
                sensor["tool_wear"],
                machine_type="L"
            )

            # 2. Convert sensor readings into RL state
            state = get_state(
                result["probability"],
                sensor["torque"],
                sensor["tool_wear"]
            )

            if state not in q_table:
                q_table[state] = [0.0] * len(ACTIONS)

            # 3. Select action
            action_index = choose_action(
                q_table,
                state,
                epsilon
            )

            action_name = ACTIONS[action_index]

            # 4. Calculate reward
            reward = get_reward(state, action_index)

            episode_reward += reward

            # 5. Apply action to simulator
            response = machine.apply_maintenance_action(
                action_name
            )

            # 6. Get next state
            if not machine.machine_running:

                next_state = state
                done = True

            else:

                next_sensor = machine.generate_sensor_data()

                next_result = predict_machine(
                    next_sensor["air_temperature"],
                    next_sensor["process_temperature"],
                    next_sensor["rotational_speed"],
                    next_sensor["torque"],
                    next_sensor["tool_wear"],
                    machine_type="L"
                )

                next_state = get_state(
                    next_result["probability"],
                    next_sensor["torque"],
                    next_sensor["tool_wear"]
                )

                done = False

            if next_state not in q_table:
                q_table[next_state] = [0.0] * len(ACTIONS)

            # 7. Q-learning update
            old_value = q_table[state][action_index]

            if done:
                next_max = 0
            else:
                next_max = max(q_table[next_state])

            new_value = old_value + ALPHA * (
                reward + GAMMA * next_max - old_value
            )

            q_table[state][action_index] = new_value

            # Stop if machine shuts down
            if done:
                break

            # Use next reading in the next iteration
            sensor = next_sensor

        total_rewards.append(episode_reward)

        # 8. Reduce exploration
        epsilon = max(
            EPSILON_MIN,
            epsilon * EPSILON_DECAY
        )

        if (episode + 1) % 50 == 0:

            average_reward = sum(
                total_rewards[-50:]
            ) / 50

            print(
                f"Episode {episode + 1:3d} | "
                f"Average Reward: {average_reward:7.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    return q_table


# ==========================================
# DISPLAY LEARNED POLICY
# ==========================================

def show_policy(q_table):

    print("\n========================================")
    print("          LEARNED Q-TABLE")
    print("========================================")

    for state, values in q_table.items():

        best_action = max(
            range(len(ACTIONS)),
            key=lambda action: values[action]
        )

        print(
            f"State {state} -> "
            f"{ACTIONS[best_action]} "
            f"| Q-values: "
            f"{[round(v, 2) for v in values]}"
        )


# ==========================================
# TEST RL AGENT
# ==========================================

def test_agent(q_table):

    print("\n========================================")
    print("       TESTING ADAPTIVE RL AGENT")
    print("========================================")

    test_modes = [
        "NORMAL",
        "MECHANICAL_FAULT",
        "OVERHEATING",
        "TOOL_WEAR"
    ]

    for fault_mode in test_modes:

        machine = MachineSimulator(
            machine_type="L",
            fault_mode=fault_mode
        )

        print("\nFault Mode:", fault_mode)

        sensor = machine.generate_sensor_data()

        for step in range(10):

            if not machine.machine_running:
                print("Machine is stopped.")
                break

            result = predict_machine(
                sensor["air_temperature"],
                sensor["process_temperature"],
                sensor["rotational_speed"],
                sensor["torque"],
                sensor["tool_wear"],
                machine_type="L"
            )

            state = get_state(
                result["probability"],
                sensor["torque"],
                sensor["tool_wear"]
            )

            # Select best learned action
            action_values = q_table.get(
                state,
                [0.0] * len(ACTIONS)
            )

            action_index = max(
                range(len(ACTIONS)),
                key=lambda a: action_values[a]
            )

            action_name = ACTIONS[action_index]

            print(f"\nStep: {step + 1}")

            print(
                "Failure Probability:",
                round(result["probability"] * 100, 2),
                "%"
            )

            print("Machine Status:", result["status"])

            print("Torque:", round(sensor["torque"], 2))

            print("Tool Wear:", round(sensor["tool_wear"], 2))

            print("RL Action:", action_name)

            # Apply action to machine
            response = machine.apply_maintenance_action(
                action_name
            )

            print("Machine Response:", response)

            if not machine.machine_running:
                break

            # Display updated readings after maintenance
            updated_sensor = machine.get_sensor_data()

            if action_name == "SCHEDULE_MAINTENANCE":

                print(
                    "Updated Torque:",
                    round(updated_sensor["torque"], 2)
                )

                print(
                    "Updated Tool Wear:",
                    round(updated_sensor["tool_wear"], 2)
                )

            sensor = machine.generate_sensor_data()


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    # Train agent
    q_table = train_agent()

    # Display learned policy
    show_policy(q_table)

    # Test agent
    test_agent(q_table)

    # Save Q-table
    output_file = (
        PROJECT_ROOT
        / "reinforcement_learning"
        / "q_table.pkl"
    )

    with open(output_file, "wb") as file:
        pickle.dump(q_table, file)

    print("\n========================================")
    print("     Q-TABLE SAVED SUCCESSFULLY")
    print("========================================")

    print(output_file)