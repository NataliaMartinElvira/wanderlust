#!/usr/bin/env python3
"""
Refactored obstacle step analyzer with safer serial I/O, real rotation integration,
throttled EXCEL saving, and improved calibration handling.

MODIFICACIÓN: Eliminada la verificación MIN_STEP_DURATION_S para centrarse SOLO en Altura/Amplitud.
"""

import argparse
import serial
import time
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import json
import threading

# ==========================================
# 1. CONFIGURATION (tweak as needed)
# ==========================================
DEFAULT_SERIAL_PORT = "COM7"
DEFAULT_BAUD_RATE = 115200

CALIBRATION_DURATION_S = 3.0
SAVE_INTERVAL_S = 5.0        # Save excel every N seconds
OUTPUT_DIR = Path("sessions")

# --- REAL DIMENSIONS (cm)
OBSTACLE_HEIGHT_CM = 6.0
OBSTACLE_DEPTH_CM = 6.0
SAFETY_MARGIN_CM = 1.5

# --- ANTI-CHEAT FILTERS ---
MIN_STEP_DURATION_S = 0.6  # Se mantiene la variable pero NO se usa en la lógica de fallo
MAX_STEP_DURATION_S = 3.5

# Detection Sensitivity (Start/Stop)
GYRO_START_THR_DPS = 15.0
GYRO_STOP_THR_DPS = 5.0

# Excel Sheet Names
SHEET_RAW = "Raw_Data"
SHEET_STEPS = "Step_Analysis"

# HEADERS (must match the V1_ACCEL packet structure)
HEADERS = [
    "time_ms", "acc_x_g", "acc_y_g", "acc_z_g",
    "pitch_deg", "roll_deg", "gyr_x_dps", "gyr_y_dps", "gyr_z_dps"
]

LEN_V1 = 9  # V1 sends 9 items (time_ms + 8 data points)


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
        try:
            self.samples.append({
                "acc_x_g": float(row["acc_x_g"]),
                "acc_y_g": float(row["acc_y_g"]),
                "acc_z_g": float(row["acc_z_g"]),
            })
        except Exception:
            pass

    def calculate(self):
        if not self.samples:
            return False
        df = pd.DataFrame(self.samples)
        if df.isnull().any().any():
            return False
        self.gravity_vector = np.array([
            df["acc_x_g"].mean(),
            df["acc_y_g"].mean(),
            df["acc_z_g"].mean(),
        ])
        return True

    def get_dynamic_force(self, ax, ay, az):
        current = np.array([ax, ay, az])
        dynamic = current - self.gravity_vector
        return float(np.linalg.norm(dynamic))


class ObstacleStepAnalyzer:
    def __init__(self, calibrator):
        self.calib = calibrator
        self.state = "IDLE"
        self.buffer = []  # will contain dicts with time_ms, g_mag, a_dyn, pitch
        self.start_time_local = 0.0  # local time.time() when movement detected

        # physical goals
        self.target_height = OBSTACLE_HEIGHT_CM + SAFETY_MARGIN_CM
        self.target_depth = OBSTACLE_DEPTH_CM

        self.feedback = {}

    def process_sample(self, row):
        try:
            gyr_x = float(row["gyr_x_dps"])
            gyr_y = float(row["gyr_y_dps"])
            gyr_z = float(row["gyr_z_dps"])
            acc_x = float(row["acc_x_g"])
            acc_y = float(row["acc_y_g"])
            acc_z = float(row["acc_z_g"])
            pitch = float(row.get("pitch_deg", 0.0))
            time_ms = float(row.get("time_ms", time.time() * 1000.0))
        except Exception:
            return None

        gyr_mag = float(np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2))
        acc_dyn = self.calib.get_dynamic_force(acc_x, acc_y, acc_z)
        current_time_local = time.time()

        if self.state == "IDLE":
            if gyr_mag > GYRO_START_THR_DPS:
                self.state = "MOVING"
                self.buffer = []
                self.start_time_local = current_time_local
                # add first sample
                self.buffer.append({
                    "time_ms": time_ms,
                    "g_mag": gyr_mag,
                    "a_dyn": acc_dyn,
                    "pitch": pitch
                })
        elif self.state == "MOVING":
            self.buffer.append({
                "time_ms": time_ms,
                "g_mag": gyr_mag,
                "a_dyn": acc_dyn,
                "pitch": pitch
            })
            duration = current_time_local - self.start_time_local

            # End of step detected (stillness)
            if gyr_mag < GYRO_STOP_THR_DPS and duration > 0.2:
                res = self._evaluate_quality(duration)
                self.state = "IDLE"
                return res

            # Timeout (too long => cancel)
            if duration > MAX_STEP_DURATION_S:
                self.state = "IDLE"
                return None

        return None

    def _evaluate_quality(self, duration_s):
        if not self.buffer:
            return None
        df = pd.DataFrame(self.buffer)

        # compute dt between rows using time_ms if present
        if "time_ms" in df.columns and df["time_ms"].notnull().all():
            times_s = df["time_ms"].astype(float) / 1000.0
            dt_series = times_s.diff().fillna(0.0)
            dt_series = dt_series.apply(lambda x: max(x, 0.0))
        else:
            n = max(len(df), 1)
            est_dt = duration_s / n
            dt_series = pd.Series([est_dt] * len(df))

        # --- METRICS ---
        peak_force = float(df["a_dyn"].max())
        avg_force = float(df["a_dyn"].mean())
        pitch_range = float(df["pitch"].max() - df["pitch"].min())

        # integrate rotation: sum(g_mag * dt)
        g_mag = df["g_mag"].astype(float)
        total_rotation = float((g_mag * dt_series).sum())

        # =======================================================
        #  MODIFICACIÓN: SE IGNORA EL FILTRO MIN_STEP_DURATION_S
        #  Se salta el bloque LOCK 1: TIME CONTROL.
        # =======================================================

        # --- HEIGHT/AMPLITUDE checks ---
        req_pitch = 5.0 + (self.target_height * 1.0)
        passed_flexion = (pitch_range >= req_pitch)

        req_peak_force = 0.25 + (self.target_height * 0.01)
        req_avg_force = 0.05 + (self.target_height * 0.002)
        passed_lift = (peak_force >= req_peak_force) and (avg_force >= req_avg_force)
        success_height = passed_flexion or passed_lift

        req_rotation = 15.0 + (self.target_depth * 0.6)
        success_amplitude = (total_rotation >= req_rotation)

        # El resultado final se basa SÓLO en Altura y Amplitud.
        unity_payload = {
            "step_ok": bool(success_height and success_amplitude),
            "fail_fast": False,  # <-- Siempre False, ignoramos el tiempo
            "fail_h": not bool(success_height),
            "fail_a": not bool(success_amplitude),
            "duration": duration_s
        }

        self.feedback = {
            "step_detected": True,
            "insufficient_height": not success_height,
            "insufficient_amplitude": not success_amplitude,
            "metrics": {
                "duration": duration_s,
                "peak_force": peak_force,
                "avg_force": avg_force,
                "pitch_range": pitch_range,
                "total_rot": total_rotation
            }
        }

        self._log_and_send_feedback(unity_payload, "QUALITY", self.feedback["metrics"])
        return unity_payload

    def _log_and_send_feedback(self, payload, analysis_type, metrics):
        # NOTA: analysis_type SIEMPRE será "QUALITY" en este código modificado
        # si se llega a evaluar el paso, ya que eliminamos el "TOO_FAST".
        print("\n" + "=" * 40)
        print(f"ANALYSIS RESULT (Type: QUALITY | Duration: {payload['duration']:.2f}s)")

        if payload["step_ok"]:
            print(" ✅ CORRECT STEP! (Passed)")
        else:
            print(" ❌ INCORRECT MOVEMENT")
            if payload["fail_h"]:
                print("    - Height Failure (Lift higher or hold foot up)")
            if payload["fail_a"]:
                print("    - Amplitude Failure (Step too short)")
        print("=" * 40 + "\n")

        # Structured feedback string
        feedback_str = f"UNITY_FEEDBACK:{payload['step_ok']},{payload['fail_fast']},{payload['fail_h']},{payload['fail_a']},{payload['duration']:.2f}"
        print(feedback_str)


# =========================
# 3. UTILS: I/O and save
# =========================
def setup_serial(port, baud, timeout=1.0):
    print(f"Connecting to {port} at {baud} baud...")
    try:
        s = serial.Serial(port, baud, timeout=timeout)
        print("Connected.")
        return s
    except Exception as e:
        print(f"Serial open failed: {e}")
        return None


def save_excel_safe(path: Path, raw_data, step_data):
    """
    Save raw_data and step_data to an Excel workbook using openpyxl engine.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
            pd.DataFrame(raw_data).to_excel(w, index=False, sheet_name=SHEET_RAW)
            if step_data:
                pd.DataFrame(step_data).to_excel(w, index=False, sheet_name=SHEET_STEPS)
            else:
                pd.DataFrame().to_excel(w, sheet_name=SHEET_STEPS)
        print(f"Excel saved: {path}")
    except Exception as e:
        print(f"Save failed: {e}")


def send_unity_feedback(serial_port, payload):
    """
    payload: dict returned by analyzer (unity_payload).
    This helper prints and writes to serial if available.
    """
    s = serial_port
    feedback_str = f"UNITY_FEEDBACK:{payload['step_ok']},{payload['fail_fast']},{payload['fail_h']},{payload['fail_a']},{payload['duration']:.2f}"
    # print("-> Sending to Unity (and console):", feedback_str) # Comentario para evitar doble impresión
    if s and s.is_open:
        try:
            s.write((feedback_str + "\n").encode("utf-8"))
        except Exception as e:
            print("Serial write failed:", e)


# =========================
# 4. MAIN
# =========================
def main(args):
    timestamp = now_ts()
    out_dir = Path(args.output_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"session_obstacle_V1_{timestamp}.xlsx"
    print(f"Saving to: {out_path}")

    calibrator = SensorCalibrator()
    analyzer = ObstacleStepAnalyzer(calibrator)

    raw_rows = []
    step_rows = []

    s = setup_serial(args.port, args.baud)
    if s is None:
        print("No serial connection. Exiting.")
        return

    print("\n>>> CALIBRATING (stay still) ... <<<")
    calibrating = True
    calib_start = time.time()

    last_save = time.time()

    try:
        while True:
            # read line (wait until newline)
            try:
                if s.in_waiting > 0:
                    line_bytes = s.readline()
                else:
                    time.sleep(0.005)
                    continue
            except Exception as e:
                print("Serial read error:", e)
                break

            try:
                line = line_bytes.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not line:
                continue

            # parse V1_ACCEL packets
            parts = None
            if line.startswith("V1_ACCEL:"):
                clean = line.replace("V1_ACCEL:", "")
                temp_parts = [p.strip() for p in clean.split(",")]
                if len(temp_parts) == LEN_V1:
                    parts = temp_parts

            if parts and len(parts) == LEN_V1:
                # convert to floats safely
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    continue
                row = dict(zip(HEADERS, vals))
                raw_rows.append(row)

                # calibration
                if calibrating:
                    calibrator.add_sample(row)
                    if (time.time() - calib_start) >= CALIBRATION_DURATION_S:
                        if calibrator.calculate():
                            calibrating = False
                            print("\n>>> CALIBRATION COMPLETE: GO! OVERCOME THE OBSTACLE <<<\n")
                        else:
                            calib_start = time.time()
                    continue  # skip analysis during calibration

                # analysis
                result = analyzer.process_sample(row)
                if result:
                    # Dado que se eliminó el 'fail_fast' en _evaluate_quality,
                    # el paso SIEMPRE se registra aquí (siempre es un resultado de CALIDAD).

                    metrics = analyzer.feedback.get("metrics", {}).copy()
                    metrics.update({
                        "ts": now_ts(),
                        "h_ok": not bool(result["fail_h"]),
                        "a_ok": not bool(result["fail_a"])
                    })
                    step_rows.append(metrics)

                    # always send feedback to Unity/serial
                    send_unity_feedback(s, result)

                    # periodic save
                    if time.time() - last_save >= SAVE_INTERVAL_S:
                        save_excel_safe(out_path, raw_rows, step_rows)
                        last_save = time.time()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if s and s.is_open:
            try:
                s.close()
            except Exception:
                pass
        # final save
        save_excel_safe(out_path, raw_rows, step_rows)
        print(f"Final File: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obstacle Step Analyzer (V1)")
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT, help="Serial port e.g. COM7 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD_RATE, help="Baud rate")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory to save session files")
    args = parser.parse_args()
    main(args)
