import serial
import time
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import json # Import json for potential future use or structured logs

# ==========================================
# 1. CONFIGURATION
# ==========================================
SERIAL_PORT = 'COM6'       # <--- CHECK YOUR PORT
BAUD_RATE = 115200

# --- REAL DIMENSIONS (CM) ---
OBSTACLE_HEIGHT_CM = 15.0   
OBSTACLE_DEPTH_CM  = 20.0   
SAFETY_MARGIN_CM   = 5.0    

# --- ANTI-CHEAT FILTERS ---
MIN_STEP_DURATION_S = 0.6   # Minimum 0.6 seconds in the air
MAX_STEP_DURATION_S = 3.5   # Maximum (if longer, it's considered noise)

# Detection Sensitivity (Start/Stop)
GYRO_START_THR_DPS = 15.0  
GYRO_STOP_THR_DPS  = 5.0   

# Excel Sheet Names
SHEET_RAW = "Raw_Data"
SHEET_STEPS = "Step_Analysis"

# HEADERS (Must match the V1_ACCEL packet structure)
HEADERS = [
    "time_ms", "acc_x_g", "acc_y_g", "acc_z_g",
    "pitch_deg", "roll_deg", "gyr_x_dps", "gyr_y_dps", "gyr_z_dps"
]

# V1 sends 9 items (Time + 8 data points)
LEN_V1 = 9 

def now_ts():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================
# 2. LOGIC CLASSES
# =========================

class SensorCalibrator:
    def __init__(self):
        self.samples = []
        self.gravity_vector = np.array([0.0, 0.0, 1.0]) 
        
    def add_sample(self, row):
        self.samples.append(row)

    def calculate(self):
        if not self.samples: return False
        df = pd.DataFrame(self.samples)
        # Average gravity vector
        self.gravity_vector = np.array([
            df['acc_x_g'].mean(), df['acc_y_g'].mean(), df['acc_z_g'].mean()
        ])
        return True

    def get_dynamic_force(self, ax, ay, az):
        current = np.array([ax, ay, az])
        # Subtract static gravity to get dynamic movement
        dynamic = current - self.gravity_vector
        return np.linalg.norm(dynamic)

class ObstacleStepAnalyzer:
    def __init__(self, calibrator):
        self.calib = calibrator
        self.state = "IDLE"
        self.buffer = []
        self.start_time = 0
        
        # Physical Goals
        self.target_height = OBSTACLE_HEIGHT_CM + SAFETY_MARGIN_CM
        self.target_depth  = OBSTACLE_DEPTH_CM
        
        self.feedback = {}

    def process_sample(self, row):
        # 1. Total Gyro Magnitude
        gyr_mag = np.sqrt(row['gyr_x_dps']**2 + row['gyr_y_dps']**2 + row['gyr_z_dps']**2)
        # 2. Real Force (Dynamic Acceleration)
        acc_dyn = self.calib.get_dynamic_force(row['acc_x_g'], row['acc_y_g'], row['acc_z_g'])
        # 3. Pitch
        pitch = row['pitch_deg']

        current_time = time.time()
        
        if self.state == "IDLE":
            if gyr_mag > GYRO_START_THR_DPS:
                self.state = "MOVING"
                self.buffer = []
                self.start_time = current_time
                # print(f"--> Starting movement...") # REMOVED PRINT

        elif self.state == "MOVING":
            self.buffer.append({
                'g_mag': gyr_mag, 
                'a_dyn': acc_dyn,
                'pitch': pitch
            })
            duration = current_time - self.start_time
            
            # End of step detected (stillness)
            if gyr_mag < GYRO_STOP_THR_DPS and duration > 0.2:
                res = self._evaluate_quality(duration)
                self.state = "IDLE"
                return res
            
            # Timeout
            if duration > MAX_STEP_DURATION_S: 
                self.state = "IDLE"

        return None

    def _evaluate_quality(self, duration):
        if not self.buffer: return None
        df = pd.DataFrame(self.buffer)
        
        # --- METRICS ---
        peak_force = df['a_dyn'].max()            # Max Peak
        avg_force  = df['a_dyn'].mean()           # Sustained effort (IMPORTANT)
        pitch_range = df['pitch'].max() - df['pitch'].min() 
        total_rotation = df['g_mag'].sum() * 0.02 
        
        # --- LOCK 1: TIME CONTROL ---
        if duration < MIN_STEP_DURATION_S:
            # Generate Unity Feedback Payload for TOO_FAST failure
            unity_payload = {
                "step_ok": False,
                "fail_fast": True,
                "fail_h": False,
                "fail_a": False,
                "duration": duration
            }
            self._log_and_print_feedback(unity_payload, "TOO_FAST", {"duration": duration})
            return unity_payload

        # --- LOCK 2 & 3: QUALITY CONTROL ---
        
        # Height Logic
        req_pitch = 5.0 + (self.target_height * 1.0)
        passed_flexion = (pitch_range >= req_pitch)
        
        req_peak_force = 0.25 + (self.target_height * 0.01)
        req_avg_force  = 0.05 + (self.target_height * 0.002) 
        passed_lift = (peak_force >= req_peak_force) and (avg_force >= req_avg_force)
        success_height = passed_flexion or passed_lift
        
        # Amplitude Logic
        req_rotation = 15.0 + (self.target_depth * 0.6)
        success_amplitude = (total_rotation >= req_rotation)

        # --- GENERATE FINAL UNITY PAYLOAD ---
        unity_payload = {
            "step_ok": success_height and success_amplitude,
            "fail_fast": False,
            "fail_h": not success_height,
            "fail_a": not success_amplitude,
            "duration": duration
        }
        
        # --- CONSOLE LOGGING (for developer) ---
        self.feedback = {
            "step_detected": True,
            "fail_reason": None, 
            "insufficient_height": not success_height,
            "insufficient_amplitude": not success_amplitude,
            "metrics": {
                "duration": duration,
                "peak_force": peak_force,
                "avg_force": avg_force, 
                "pitch_range": pitch_range,
                "total_rot": total_rotation
            }
        }
        
        self._log_and_print_feedback(unity_payload, "QUALITY", self.feedback["metrics"])
        
        return unity_payload

    def _log_and_print_feedback(self, payload, analysis_type, metrics):
        """Prints formatted feedback to console and sends the structured payload data string for Unity."""
        
        print("\n" + "="*40)
        print(f"ANALYSIS RESULT (Type: {analysis_type} | Duration: {payload['duration']:.2f}s)")
        
        if analysis_type == "TOO_FAST":
            print(" [!] STEP REJECTED: TOO FAST. Do it smoothly and controlled.")
        
        elif payload["step_ok"]:
            print(" ✅ CORRECT STEP! (Passed)")
        
        else:
            print(" ❌ INCORRECT MOVEMENT")
            if payload["fail_h"]: 
                print("    - Height Failure (Lift higher or hold foot up)")
            if payload["fail_a"]: 
                print("    - Amplitude Failure (Step too short)")
                
        print("="*40 + "\n")
        
        # STRUCTURED FEEDBACK OUTPUT FOR UNITY (to be read via serial/TCP connection)
        # Format: step_ok, fail_fast, fail_h, fail_a, duration
        # Example: UNITY_FEEDBACK:True,False,False,False,1.55
        print(f"UNITY_FEEDBACK:{payload['step_ok']},{payload['fail_fast']},{payload['fail_h']},{payload['fail_a']},{payload['duration']:.2f}")

# =========================
# 3. CONNECTION & MAIN
# =========================
def setup_serial():
    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    try:
        s = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Connected.")
        return s
    except Exception as e:
        print(f"Error: {e}")
        return None

def save_excel_safe(path, raw_data, step_data):
    if not raw_data: return
    try:
        with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
            pd.DataFrame(raw_data).to_excel(w, index=False, sheet_name=SHEET_RAW)
            if step_data:
                pd.DataFrame(step_data).to_excel(w, index=False, sheet_name=SHEET_STEPS)
            else:
                pd.DataFrame().to_excel(w, sheet_name=SHEET_STEPS)
        print("Excel saved.")
    except Exception as e: 
        print(f"Save failed: {e}")

def main():
    timestamp = now_ts()
    out_path = Path(__file__).parent / f"session_obstacle_V1_{timestamp}.xlsx"
    print(f"Saving to: {out_path}")
    
    calibrator = SensorCalibrator()
    analyzer = ObstacleStepAnalyzer(calibrator)
    
    raw_rows = []
    step_rows = []
    
    s = setup_serial()
    if not s: return
    
    print("\n>>> CALIBRATING (3s)... STAY STILL <<<")
    calibrating = True
    calib_start = time.time()

    try:
        while True:
            if s.in_waiting > 0:
                line = s.readline().decode("utf-8", errors="ignore").strip()
                
                # ==========================================
                # PARSING LOGIC (V1_ACCEL FILTER)
                # ==========================================
                parts = []
                
                if line.startswith("V1_ACCEL:"):
                    clean_line = line.replace("V1_ACCEL:", "")
                    temp_parts = clean_line.split(',')
                    
                    # Expect 9 items (Time + 8 data points)
                    if len(temp_parts) == LEN_V1:
                        parts = temp_parts
                
                # Only proceed if we have a valid V1 packet
                if len(parts) == LEN_V1:
                    try:
                        # Convert strings to floats
                        vals = [float(x) for x in parts]
                        # Map to dictionary for the Analyzer
                        row = dict(zip(HEADERS, vals))
                        
                        # Store Raw Data
                        raw_rows.append(row)
                        
                        # --- LOGIC: CALIBRATION PHASE ---
                        if calibrating:
                            calibrator.add_sample(row)
                            if time.time() - calib_start > 3.0:
                                if calibrator.calculate():
                                    calibrating = False
                                    print("\n>>> GO! OVERCOME THE OBSTACLE WITH CONTROL <<<\n")
                                else:
                                    calib_start = time.time()
                            continue # Skip analysis while calibrating

                        # --- LOGIC: ANALYSIS PHASE ---
                        result = analyzer.process_sample(row)
                        
                        if result:
                            # If step was rejected for being too fast
                            if result.get("fail_reason") == "TOO_FAST":
                                # The feedback function handles printing the rejection message and the UNITY_FEEDBACK string
                                pass 
                            else:
                                log = result["metrics"]
                                log.update({
                                    "ts": now_ts(), 
                                    "h_ok": not result["insufficient_height"], 
                                    "a_ok": not result["insufficient_amplitude"]
                                })
                                step_rows.append(log)
                                
                                # Save periodically on successful steps
                                save_excel_safe(out_path, raw_rows, step_rows)

                    except ValueError:
                        continue

    except KeyboardInterrupt:
        print("\nFinished.")
    finally:
        if 's' in locals() and s.is_open: s.close()
        save_excel_safe(out_path, raw_rows, step_rows)
        print(f"Final File: {out_path}")

if __name__ == "__main__":
    main()