import serial
import time
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
SERIAL_PORT = 'COM7'
BAUD_RATE = 115200

# --- DIMENSIONES REALES DEL OBSTÁCULO ---
OBSTACLE_HEIGHT_CM = 6.0
OBSTACLE_DEPTH_CM  = 6.0
SAFETY_MARGIN_CM   = 5.0

# --- FILTROS TÉCNICOS ---
GYRO_START_THR_DPS = 15.0
GYRO_STOP_THR_DPS  = 5.0
MAX_STEP_DURATION_S = 4.0

# Excel
SHEET_RAW = "Raw_Data"
SHEET_STEPS = "Step_Analysis"

HEADERS = [
    "time_ms", "acc_x_g", "acc_y_g", "acc_z_g",
    "pitch_deg", "roll_deg", "gyr_x_dps", "gyr_y_dps", "gyr_z_dps"
]
LEN_V1 = 9

def now_ts():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================
# 2. LÓGICA BIOMECÁNICA MEJORADA
# =========================

class SensorCalibrator:
    def __init__(self):
        self.samples = []
        self.gravity_vector = np.array([0.0, 0.0, 1.0])

    def add_sample(self, row):
        self.samples.append(row)

    def calculate(self):
        if not self.samples:
            return False
        df = pd.DataFrame(self.samples)
        self.gravity_vector = np.array([
            df['acc_x_g'].mean(),
            df['acc_y_g'].mean(),
            df['acc_z_g'].mean()
        ])
        return True

    def get_dynamic_force(self, ax, ay, az):
        current = np.array([ax, ay, az])
        dynamic = current - self.gravity_vector
        return np.linalg.norm(dynamic)


class ObstacleStepAnalyzer:
    def __init__(self, calibrator):
        self.calib = calibrator
        self.state = "IDLE"
        self.buffer = []
        self.start_time = 0

        self.target_height = OBSTACLE_HEIGHT_CM + SAFETY_MARGIN_CM
        self.target_depth  = OBSTACLE_DEPTH_CM

        self.feedback = {}

    def process_sample(self, row):
        # giroscopio corregido (**2)
        gyr_mag = np.sqrt(
            row['gyr_x_dps']**2 +
            row['gyr_y_dps']**2 +
            row['gyr_z_dps']**2
        )

        acc_dyn = self.calib.get_dynamic_force(
            row['acc_x_g'], row['acc_y_g'], row['acc_z_g']
        )

        current_time = time.time()

        if self.state == "IDLE":
            if gyr_mag > GYRO_START_THR_DPS:
                self.state = "MOVING"
                self.buffer = []
                self.start_time = current_time

        elif self.state == "MOVING":
            self.buffer.append({
                'g_mag': gyr_mag,
                'a_dyn': acc_dyn,
                'ax': row['acc_x_g'],
                'ay': row['acc_y_g'],
                'az': row['acc_z_g']
            })

            duration = current_time - self.start_time

            if gyr_mag < GYRO_STOP_THR_DPS and duration > 0.2:
                res = self._evaluate_quality(duration)
                self.state = "IDLE"
                return res

            if duration > MAX_STEP_DURATION_S:
                self.state = "IDLE"

        return None

    def _evaluate_quality(self, duration):
        if not self.buffer:
            return None

        df = pd.DataFrame(self.buffer)

        if duration < 0.6:
            return None

        # ======================================
        # NUEVA LÓGICA BIOMECÁNICA ROBUSTA
        # ======================================

        # Señal real de levantamiento del pie
        df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
        lift_signal = df["acc_mag"].max() - df["acc_mag"].min()

        avg_force = df['a_dyn'].mean()
        total_rotation = df['g_mag'].sum() * 0.02

        # Requisito de levantamiento (altura)
        req_lift = 0.08 + (self.target_height * 0.01)
        passed_height = lift_signal >= req_lift

        # Amplitud SOLO si hubo altura
        if passed_height:
            req_rotation = 15.0 + (self.target_depth * 0.6)
            passed_amplitude = total_rotation >= req_rotation
        else:
            passed_amplitude = False

        self.feedback = {
            "step_detected": True,
            "insufficient_height": not passed_height,
            "insufficient_amplitude": not passed_amplitude,
            "metrics": {
                "duration": duration,
                "lift_signal": lift_signal,
                "avg_force": avg_force,
                "total_rot": total_rotation
            }
        }

        # ----------- SALIDA POR CONSOLA -----------
        print("\n" + "="*40)
        print(f"Step analysis (Duration: {duration:.2f}s)")

        if passed_height and passed_amplitude:
            print(" Correct step")
        else:
            print(" Wrong Movement")
            if not passed_height:
                print("  - Height error: LIFT YOUR FOOT HIGHER!")
            elif not passed_amplitude:
                print("  - Amplitude error: STEP TOO SHORT!")

        print("="*40 + "\n")

        return self.feedback


# =========================
# 3. MAIN (SERIAL)
# =========================

def setup_serial():
    print(f"Conected to {SERIAL_PORT}...")
    try:
        s = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Conected.")
        return s
    except Exception as e:
        print(f"Error Serial: {e}")
        return None

def save_excel_safe(path, raw_data, step_data):
    if not raw_data:
        return
    try:
        with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
            pd.DataFrame(raw_data).to_excel(w, index=False, sheet_name=SHEET_RAW)
            if step_data:
                pd.DataFrame(step_data).to_excel(w, index=False, sheet_name=SHEET_STEPS)
            else:
                pd.DataFrame().to_excel(w, sheet_name=SHEET_STEPS)
    except:
        pass


def main():
    timestamp = now_ts()
    out_path = Path.cwd() / f"session_obstacle_FINAL{timestamp}.xlsx"
    print(f"Saved in: {out_path}")

    calibrator = SensorCalibrator()
    analyzer = ObstacleStepAnalyzer(calibrator)

    raw_rows = []
    step_rows = []

    s = setup_serial()
    if not s:
        return

    print("\n>>> CALIBRATING (3s)... STAY STILL <<<")
    calibrating = True
    calib_start = time.time()

    try:
        while True:
            if s.in_waiting > 0:
                line = s.readline().decode("utf-8", errors="ignore").strip()
                parts = []

                if line.startswith("V1_ACCEL:"):
                    clean_line = line.replace("V1_ACCEL:", "")
                    parts = clean_line.split(',')

                if len(parts) == LEN_V1:
                    try:
                        vals = [float(x) for x in parts]
                        row = dict(zip(HEADERS, vals))
                        raw_rows.append(row)

                        if calibrating:
                            calibrator.add_sample(row)
                            if time.time() - calib_start > 3.0:
                                if calibrator.calculate():
                                    calibrating = False
                                    print("\n>>> READY! GET OVER THE OBSTACLE <<<\n")
                                else:
                                    calib_start = time.time()
                            continue

                        result = analyzer.process_sample(row)
                        if result:
                            log = result["metrics"]
                            log.update({
                                "ts": now_ts(),
                                "h_ok": not result["insufficient_height"],
                                "a_ok": not result["insufficient_amplitude"]
                            })
                            step_rows.append(log)
                            save_excel_safe(out_path, raw_rows, step_rows)

                    except ValueError:
                        continue

    except KeyboardInterrupt:
        print("\nFin.")

    finally:
        if 's' in locals() and s.is_open:
            s.close()
        save_excel_safe(out_path, raw_rows, step_rows)


if __name__ == "__main__":
    main()
