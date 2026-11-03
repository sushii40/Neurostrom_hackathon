import random


class MachineSimulator:

    def __init__(self, machine_type="L", fault_mode="NORMAL"):

        self.machine_type = machine_type
        self.fault_mode = fault_mode

        # Initial machine values
        self.air_temperature = 300.0
        self.process_temperature = 310.0
        self.rotational_speed = 1500.0
        self.torque = 40.0
        self.tool_wear = 50.0

        # Machine operating state
        self.machine_running = True
        self.monitoring_level = "NORMAL"
        self.maintenance_count = 0

    # ==========================================
    # GENERATE SENSOR DATA
    # ==========================================

    def generate_sensor_data(self):

        # Stop updating readings after shutdown
        if not self.machine_running:
            return self.get_sensor_data()

        # Normal machine behaviour
        self.air_temperature += random.uniform(-0.5, 0.5)
        self.process_temperature += random.uniform(-0.5, 0.5)
        self.rotational_speed += random.uniform(-20, 20)
        self.torque += random.uniform(-2, 2)
        self.tool_wear += random.uniform(0.1, 0.5)

        # Fault injection
        if self.fault_mode == "OVERHEATING":

            self.air_temperature += random.uniform(1.0, 2.0)
            self.process_temperature += random.uniform(2.0, 4.0)

        elif self.fault_mode == "MECHANICAL_FAULT":

            self.torque += random.uniform(3.0, 6.0)
            self.rotational_speed += random.uniform(-80, 80)

        elif self.fault_mode == "TOOL_WEAR":

            self.tool_wear += random.uniform(3.0, 6.0)
            self.torque += random.uniform(1.0, 3.0)

        # Keep values within dataset range
        self.air_temperature = max(
            295, min(self.air_temperature, 320)
        )

        self.process_temperature = max(
            305, min(self.process_temperature, 330)
        )

        self.rotational_speed = max(
            1000, min(self.rotational_speed, 2000)
        )

        self.torque = max(
            20, min(self.torque, 70)
        )

        self.tool_wear = min(
            self.tool_wear, 250
        )

        return self.get_sensor_data()

    # ==========================================
    # GET CURRENT SENSOR DATA
    # ==========================================

    def get_sensor_data(self):

        return {
            "air_temperature": self.air_temperature,
            "process_temperature": self.process_temperature,
            "rotational_speed": self.rotational_speed,
            "torque": self.torque,
            "tool_wear": self.tool_wear,
            "machine_type": self.machine_type,
            "machine_running": self.machine_running
        }

    # ==========================================
    # APPLY RL MAINTENANCE ACTION
    # ==========================================

    def apply_maintenance_action(self, action):

        if action == "CONTINUE_OPERATION":

            self.monitoring_level = "NORMAL"

            return "Machine continues normal operation."

        elif action == "MONITOR":

            self.monitoring_level = "HIGH"

            return "Monitoring frequency increased."

        elif action == "SCHEDULE_MAINTENANCE":

            # Simulate maintenance and component servicing
            self.air_temperature = max(
                295, self.air_temperature - 3
            )

            self.process_temperature = max(
                305, self.process_temperature - 5
            )

            self.rotational_speed = 1500.0

            self.torque = max(
                20, self.torque * 0.75
            )

            # Simulate replacement or servicing of worn tooling
            self.tool_wear = min(
                self.tool_wear, 20.0
            )

            self.fault_mode = "NORMAL"
            self.machine_running = True
            self.monitoring_level = "NORMAL"

            self.maintenance_count += 1

            return "Maintenance completed. Machine condition improved."

        elif action == "EMERGENCY_SHUTDOWN":

            self.machine_running = False

            return "Emergency shutdown activated. Machine stopped."

        else:

            return "Invalid maintenance action."