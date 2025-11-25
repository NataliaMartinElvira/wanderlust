import time

class ExerciseLogicTemplate:
    """
    GENERIC TEMPLATE: Placeholder for specific exercise logic (e.g., SeatedMarchLogic).
    Used to define the interface required by imu_main_controller.py.
    """
    def __init__(self):
        print("[Logic] Generic exercise logic template loaded.")

    def check_calmness(self, raw_data_dict):
        """ Checks for required stillness for calibration. Returns True when calibrated. """
        if not raw_data_dict:
            return False
        # Placeholder logic: waits for 3 seconds of 'quiet' readings (simulated here)
        return time.time() % 10 < 3 # Simulating quick calibration for other exercises

    def analyze_performance(self, raw_data_dict):
        """ 
        Analyzes IMU data during the exercise.
        Returns: True if negative feedback is needed (FEEDBACK:BAD), False otherwise (FEEDBACK:GOOD).
        """
        # Placeholder logic: Simulating intermittent bad performance
        return time.time() % 8 < 2 # True for 2 seconds every 8 seconds cycle