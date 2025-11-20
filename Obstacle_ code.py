import socket
import time
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 1. ADVANCED CONFIGURATION
# ==========================================
HOST = "192.168.4.1"
PORT = 3333

# --- REAL DIMENSIONS (CM) ---
OBSTACLE_HEIGHT_CM = 15.0   
OBSTACLE_DEPTH_CM  = 20.0   
SAFETY_MARGIN_CM   = 5.0    

# --- ANTI-CHEAT FILTERS ---
# A real step to overcome an object requires control.
# If it lasts less than this, it is a kick or a spasm.
MIN_STEP_DURATION_S = 0.6   # Minimum 0.6 seconds in the air
MAX_STEP_DURATION_S = 3.5   # Maximum (if longer, it's considered noise)

# Detection Sensitivity (Start/Stop)
GYRO_START_THR_DPS = 15.0  # Increased slightly to ignore tremors
GYRO_STOP_THR_DPS  = 5.0   

# Excel Sheet Names
SHEET_RAW = "Raw_Data"
SHEET_STEPS = "Step_Analysis"

HEADERS = [
    "time_ms", "acc_x_g", "acc_y_g", "acc_z_g",
    "pitch_deg", "roll_deg", "gyr_x_dps", "gyr_y_dps", "gyr_z_dps"
]

def now_ts():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================
# 2. LOGIC
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
                print(f"--> Starting movement...")

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
        # If it was too fast, it is a spasm or kick, not a controlled step.
        if duration < MIN_STEP_DURATION_S:
            return {
                "step_detected": True,
                "fail_reason": "TOO_FAST",
                "metrics": {"duration": duration, "peak_force": peak_force}
            }

        # --- LOCK 2: HEIGHT (Improved Hybrid Logic) ---
        
        # A) Via Knee Flexion
        # Pitch required: 1 degree per cm (softer requirement)
        req_pitch = 5.0 + (self.target_height * 1.0)
        passed_flexion = (pitch_range >= req_pitch)
        
        # B) Via Block Lifting (Hip Flexion)
        # We require Peak Force AND Sustained Average Force.
        req_peak_force = 0.25 + (self.target_height * 0.01)
        req_avg_force  = 0.05 + (self.target_height * 0.002) 
        
        passed_lift = (peak_force >= req_peak_force) and (avg_force >= req_avg_force)
        
        success_height = passed_flexion or passed_lift
        
        # --- LOCK 3: AMPLITUDE (Step Length) ---
        req_rotation = 15.0 + (self.target_depth * 0.6)
        success_amplitude = (total_rotation >= req_rotation)

        # --- FINAL RESULT ---
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
        
        # --- CONSOLE PRINT ---
        print("\n" + "="*40)
        print(f"QUALITY ANALYSIS (Duration: {duration:.2f}s)")
        
        if passed_flexion:
            print(f" > Strategy: KNEE FLEXION (Pitch: {pitch_range:.1f} deg)")
        elif passed_lift:
             print(f" > Strategy: HIP STRENGTH (Sustained Force)")
        else:
             print(f" > Strategy: NOT DETECTED (Insufficient movement)")
             print(f"   (Pitch: {pitch_range:.1f} deg | PeakG: {peak_force:.2f}g | AvgG: {avg_force:.3f}g)")

        if success_height and success_amplitude:
            print("\n ✅ CORRECT STEP! (Passed)")
        else:
            print("\n ❌ INCORRECT MOVEMENT")
            if not success_height: print("    - Height Failure (Lift higher or hold foot up)")
            if not success_amplitude: print("    - Amplitude Failure (Step too short)")
            
        print("="*40 + "\n")

        return self.feedback

# =========================
# 3. CONNECTION & MAIN
# =========================
def connect():
    while True:
        try:
            print(f"Connecting to {HOST}:{PORT} ...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((HOST, PORT))
            s.settimeout(None)
            print("Connected.")
            return s
        except Exception:
            time.sleep(2)

def save_excel_safe(path, raw_data, step_data):
    if not raw_data: return
    try:
        with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
            pd.DataFrame(raw_data).to_excel(w, index=False, sheet_name=SHEET_RAW)
            if step_data:
                pd.DataFrame(step_data).to_excel(w, index=False, sheet_name=SHEET_STEPS)
            else:
                pd.DataFrame().to_excel(w, sheet_name=SHEET_STEPS)
    except: pass

def main():
    timestamp = now_ts()
    out_path = Path(__file__).parent / f"session_anti_cheat_{timestamp}.xlsx"
    
    calibrator = SensorCalibrator()
    analyzer = ObstacleStepAnalyzer(calibrator)
    
    raw_rows = []
    step_rows = []
    buffer_str = ""
    
    s = connect()
    
    print("\n>>> CALIBRATING (3s)... STAY STILL <<<")
    calibrating = True
    calib_start = time.time()

    try:
        while True:
            try:
                data = s.recv(2048)
                if not data: break
                buffer_str += data.decode("utf-8", errors="ignore")
            except: break

            while "\n" in buffer_str:
                line, buffer_str = buffer_str.split("\n", 1)
                parts = [p.strip() for p in line.split(",")]

                if len(parts) == len(HEADERS) and parts[0] != "time_ms":
                    try:
                        vals = [float(x) for x in parts]
                        row = dict(zip(HEADERS, vals))
                        raw_rows.append(row)
                        
                        if calibrating:
                            calibrator.add_sample(row)
                            if time.time() - calib_start > 3.0:
                                if calibrator.calculate():
                                    calibrating = False
                                    print("\n>>> GO! OVERCOME THE OBSTACLE WITH CONTROL <<<\n")
                                else:
                                    calib_start = time.time()
                            continue

                        result = analyzer.process_sample(row)
                        if result:
                            # If step was rejected for being too fast
                            if result.get("fail_reason") == "TOO_FAST":
                                print(f" [!] TOO FAST! ({result['metrics']['duration']:.2f}s)")
                                print("     Do it smoothly and controlled. No kicking.\n")
                            else:
                                log = result["metrics"]
                                log.update({
                                    "ts": now_ts(), 
                                    "h_ok": not result["insufficient_height"], 
                                    "a_ok": not result["insufficient_amplitude"]
                                })
                                step_rows.append(log)

                    except ValueError:
                        continue
    except KeyboardInterrupt:
        print("\nFinished.")
    finally:
        try: s.close()
        except: pass
        save_excel_safe(out_path, raw_rows, step_rows)

if __name__ == "__main__":
    main()