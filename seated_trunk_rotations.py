# seated_trunk_rotation_analysis.py
# Reads two-IMU stream from ESP32, fuses with Madgwick, computes trunk-vs-pelvis axial rotation,
# detects reps & reports performance metrics for seated trunk rotations.
# does not require old “imu_to_excel_wireless” logger—the script connects to the ESP32 directly and works wirelessly by itself

# how to run: Power ESP32, join Wi-Fi IMU_Logger (pass imu12345). python seated_trunk_rotation_analysis.py; Ask the patient to keep still for 3 s (calibration), then perform slow L/R rotations. Press Ctrl+C to end; see trunk_rotation_session.csv and on-screen summary.

# trunk_rotation_feedback.py
# Real-time feedback for seated trunk rotations using two IMUs (upper trunk & pelvis).
# - Visual: terminal status + ASCII bars with color
# - Audio: system bell for cues (cross-platform '\a')
#
# Requires: pip install ahrs numpy pandas scipy
# Connect to ESP32 AP "IMU_Logger" (pass: imu12345) before running.

import socket
import time
import sys
import numpy as np
from ahrs.filters import Madgwick
from scipy.signal import savgol_filter

HOST = "192.168.4.1"
PORT = 3333

HEADERS = ["imu_id","time_ms","acc_x_g","acc_y_g","acc_z_g",
           "pitch_deg","roll_deg","gyr_x_dps","gyr_y_dps","gyr_z_dps"]

# ------- Tunables -------
SAMPLE_HZ               = 10.0
CALIBRATION_SECONDS     = 3.0

# Task goal
YAW_TARGET_DEG          = 35.0    # target amplitude each side
YAW_HYST_DEG            = 5.0     # to avoid chattering

# Clean movement rules
MAX_YAW_SPEED_DPS       = 120.0   # speed ceiling; > means "too fast"
MAX_COMP_ANGLE_DEG      = 10.0    # |relative pitch| or |relative roll| during rotation
MAX_PELVIS_DRIFT_DEG    = 10.0    # absolute pelvis yaw deviation from neutral

# Rep logic
MIN_REP_GAP_S           = 1.0     # minimum time between peaks
SMOOTH_WINDOW           = 5       # small smoothing for velocity; odd integer >= 3
SMOOTH_POLY             = 2
ASYM_WINDOW_REPS        = 4       # rolling window to assess asymmetry
ASYM_ALERT_DIFF_DEG     = 10.0    # mean L vs R difference to alert

UPPER_ID  = "IMU_CH0"
PELVIS_ID = "IMU_CH3"

# ------- Utility -------
def beep():
    # cross-platform minimal beep
    print("\a", end="", flush=True)

def color(txt, c):
    codes = {
        "red":"\033[31m", "green":"\033[32m", "yellow":"\033[33m",
        "blue":"\033[34m", "mag":"\033[35m", "cyan":"\033[36m",
        "bold":"\033[1m", "reset":"\033[0m"
    }
    return f"{codes.get(c,'')}{txt}{codes['reset']}"

def bar(value, span=60, max_abs=60.0):
    # ASCII bar centered at 0 for yaw display
    value = float(np.clip(value, -max_abs, max_abs))
    half = span//2
    pos = int(round((value/max_abs)*half))
    left = "-"*(half+min(0,pos))
    mid  = "|"
    right= "-"*(half-max(pos,0))
    return f"[{left}{mid}{right}] {value:+.1f}°"

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def euler_zyx(q):
    w,x,y,z = q
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    sinp  = 2*(w*y - z*x); sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    return roll, pitch, yaw

def connect():
    while True:
        try:
            print(f"Connecting to {HOST}:{PORT} …")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect((HOST, PORT))
            s.settimeout(None)
            print("Connected.\n")
            return s
        except Exception as e:
            print(f"Connect failed: {e}. Retrying in 2s…")
            time.sleep(2)

def parse_line(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(HEADERS):
        return None
    try:
        imu_id = parts[0]
        t_ms   = int(float(parts[1]))
        ax,ay,az = map(float, parts[2:5])
        gx,gy,gz = map(float, parts[7:10])  # deg/s
        gx,gy,gz = np.radians([gx,gy,gz])   # rad/s for filter
        return imu_id, t_ms, np.array([ax,ay,az]), np.array([gx,gy,gz])
    except Exception:
        return None

# ------- Main -------
def main():
    print(color("Seated trunk rotation – Real-time feedback", "bold"))
    print("Keep still for ~3 s at start (neutral calibration).")
    print(f"Targets: |yaw| ≥ {YAW_TARGET_DEG:.0f}°, speed ≤ {MAX_YAW_SPEED_DPS:.0f}°/s, "
          f"comp ≤ {MAX_COMP_ANGLE_DEG:.0f}°, pelvis drift ≤ {MAX_PELVIS_DRIFT_DEG:.0f}°.\n")

    # Filters
    f_upper = Madgwick(beta=0.1, frequency=SAMPLE_HZ)
    f_pel   = Madgwick(beta=0.1, frequency=SAMPLE_HZ)
    q_upper = np.array([1.,0.,0.,0.])
    q_pel   = np.array([1.,0.,0.,0.])

    # Pelvis yaw neutral (absolute) & relative yaw zeroing
    calib_start = None
    yaw_rel_bias = None
    pel_yaw_bias = None
    collected_rel_yaw = []
    collected_pel_yaw = []

    # Rep tracking
    last_time = None
    last_yaw = None
    last_peak_time = -1e9
    peaks_L = []  # negative yaw
    peaks_R = []  # positive yaw

    # Smoothing buffers
    yaw_buf = []
    time_buf = []

    # State flags
    in_pos_zone = False
    in_neg_zone = False

    # Socket
    s = connect()
    buffer = ""
    header_seen = False
    start_ms = None

    try:
        while True:
            data = s.recv(1024)
            # DEBUG: show that bytes are arriving
            if data:
                print(f"\rReceived {len(data)} bytes", end="")
            if not data:
                raise ConnectionError("Remote closed")
            buffer += data.decode("utf-8", errors="ignore")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                parts = [p.strip() for p in line.split(",")]
                if not header_seen:
                    if parts == HEADERS:
                        header_seen = True
                        continue
                    header_seen = True  # joined mid-stream

                parsed = parse_line(line)
                if parsed is None:
                    # print the first few unparsed lines to see the format
                    print(f"\nUNPARSED: {line}")
                    continue
                imu_id, t_ms, acc, gyr = parsed
                if start_ms is None:
                    start_ms = t_ms
                t = (t_ms - start_ms) / 1000.0

                # Update filters
                if imu_id == UPPER_ID:
                    q_upper = f_upper.updateIMU(gyr=gyr, acc=acc, q=q_upper)
                elif imu_id == PELVIS_ID:
                    q_pel = f_pel.updateIMU(gyr=gyr, acc=acc, q=q_pel)
                else:
                    continue

                # Need both
                if q_upper is None or q_pel is None:
                    continue

                # Relative & absolute angles
                q_rel = quat_mul(q_upper, quat_conj(q_pel))
                r_rel, p_rel, y_rel = euler_zyx(q_rel)
                r_pel, p_pel, y_pel = euler_zyx(q_pel)

                # Calibration phase
                if calib_start is None:
                    calib_start = t

                if (t - calib_start) <= CALIBRATION_SECONDS:
                    collected_rel_yaw.append(y_rel)
                    collected_pel_yaw.append(y_pel)

                    msg = f"Calibrating... {t - calib_start:0.1f}/{CALIBRATION_SECONDS:.0f}s"
                    print(msg.ljust(80), end="\r")
                    continue

                if yaw_rel_bias is None:
                    yaw_rel_bias = float(np.median(collected_rel_yaw)) if collected_rel_yaw else 0.0
                    pel_yaw_bias = float(np.median(collected_pel_yaw)) if collected_pel_yaw else 0.0
                    print(" " * 80, end="\r")
                    print(color(f"Calibration done. Rel yaw bias {yaw_rel_bias:+.1f}°, pelvis yaw bias {pel_yaw_bias:+.1f}°.", "green"))

                # Bias-corrected values
                y_rel -= yaw_rel_bias
                y_pel_dev = y_pel - pel_yaw_bias  # pelvis absolute deviation

                # Smoothing for velocity
                yaw_buf.append(y_rel); time_buf.append(t)
                if len(yaw_buf) >= max(3, SMOOTH_WINDOW):
                    if len(yaw_buf) >= SMOOTH_WINDOW and SMOOTH_WINDOW % 2 == 1:
                        y_smooth = savgol_filter(np.array(yaw_buf[-SMOOTH_WINDOW:]), SMOOTH_WINDOW, SMOOTH_POLY)[-1]
                    else:
                        y_smooth = yaw_buf[-1]
                else:
                    y_smooth = y_rel

                if len(time_buf) >= 2:
                    dt = time_buf[-1] - time_buf[-2]
                    dy = (y_smooth - (y_smooth if last_yaw is None else last_yaw))
                    yaw_vel = dy / dt if dt > 0 else 0.0
                else:
                    dt = 0.0
                    dy = 0.0
                    yaw_vel = 0.0

                last_yaw = y_smooth

                # --- Rule checks ---
                errors = []
                warnings = []

                # Speed
                if abs(yaw_vel) > np.radians(MAX_YAW_SPEED_DPS):  # yaw_vel currently in deg/s? wait we computed using deg
                    pass
                # NOTE: We computed y_smooth in degrees; dy is degrees; dt in seconds => deg/s.
                # So compare directly:
                if abs(dy / (dt if dt>0 else 1)) > MAX_YAW_SPEED_DPS:
                    errors.append("Too fast – slow down")

                # Compensation during rotation phase (when moving or in target)
                in_rotation = (abs(y_smooth) > 15.0) or (abs(dy) > 2.0)
                if in_rotation and (abs(p_rel) > MAX_COMP_ANGLE_DEG or abs(r_rel) > MAX_COMP_ANGLE_DEG):
                    errors.append("Avoid bending/leaning (keep chest upright)")

                # Pelvis drift
                if abs(y_pel_dev) > MAX_PELVIS_DRIFT_DEG:
                    errors.append("Keep hips still")

                # Target encouragement
                if abs(y_smooth) < YAW_TARGET_DEG and in_rotation:
                    warnings.append("Rotate further")

                # Rep peaks + asymmetry (simple hysteresis)
                up_th, down_th = YAW_TARGET_DEG, -YAW_TARGET_DEG
                exit_pos, exit_neg = up_th - YAW_HYST_DEG, down_th + YAW_HYST_DEG

                # Zones
                if not in_pos_zone and y_smooth >= up_th and (t - last_peak_time) >= MIN_REP_GAP_S:
                    in_pos_zone = True; in_neg_zone = False
                    peaks_R.append((t, y_smooth)); last_peak_time = t
                    beep()
                    print(color("✓ Target reached (Right)", "green"))

                if not in_neg_zone and y_smooth <= down_th and (t - last_peak_time) >= MIN_REP_GAP_S:
                    in_neg_zone = True; in_pos_zone = False
                    peaks_L.append((t, y_smooth)); last_peak_time = t
                    beep()
                    print(color("✓ Target reached (Left)", "green"))

                if in_pos_zone and y_smooth < exit_pos:
                    in_pos_zone = False
                if in_neg_zone and y_smooth > exit_neg:
                    in_neg_zone = False

                # Asymmetry check (rolling window)
                def mean_last(arr, n):
                    if not arr: return None
                    vals = [v for (_,v) in arr[-n:]]
                    return float(np.mean(vals)) if vals else None
                mL = mean_last(peaks_L, ASYM_WINDOW_REPS)
                mR = mean_last(peaks_R, ASYM_WINDOW_REPS)
                asym_msg = ""
                if mL is not None and mR is not None and abs(mL - mR) > ASYM_ALERT_DIFF_DEG:
                    asym_msg = f"Asymmetry: L={mL:.0f}° R={mR:.0f}° – try to match both sides"
                    # treat as coaching cue, not hard error

                # --- UI output ---
                # One-line status
                bar_txt = bar(y_smooth, span=60, max_abs=max(60.0, YAW_TARGET_DEG*1.3))
                status = []
                if errors:
                    status.append(color(" | ".join(errors), "red"))
                if warnings and not errors:
                    status.append(color(" | ".join(warnings), "yellow"))
                if asym_msg and not errors:
                    status.append(color(asym_msg, "mag"))

                status_line = f"{bar_txt} | vel { (dy/(dt if dt>0 else 1)):+5.0f}°/s | pelvis {y_pel_dev:+.1f}°"
                if status:
                    status_line += " | " + " | ".join(status)

                print(status_line.ljust(140), end="\r")

                # Error beep (only once per update when any error appears)
                if errors:
                    beep()

    except KeyboardInterrupt:
        print("\nStopping…")
    except (ConnectionError, OSError) as e:
        print(f"\nConnection error: {e}")
    finally:
        try:
            s.close()
        except Exception:
            pass

        # Session summary
        nL, nR = len(peaks_L), len(peaks_R)
        print(color(f"\nSession summary: peaks L={nL}, R={nR}", "bold"))
        if peaks_L:
            print(f"  Left mean peak:  {np.mean([v for _,v in peaks_L]):.1f}°")
        if peaks_R:
            print(f"  Right mean peak: {np.mean([v for _,v in peaks_R]):.1f}°")
        if peaks_L and peaks_R:
            print(f"  Asymmetry (R-L): {np.mean([v for _,v in peaks_R]) - np.mean([v for _,v in peaks_L]):+.1f}°")
        print("Done.")
        

if __name__ == "__main__":
    main()
