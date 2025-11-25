# measure_trunk_rom_axial.py
#
# Measure patient's seated trunk rotation range of motion (ROM) using two IMUs.
# - Same assumptions as the main trunk-rotation script:
#   * UPPER_ID  = thorax IMU
#   * PELVIS_ID = pelvis IMU
#   * IMU +X ≈ spine up
# - 3 s calibration defines the patient's personal "upright" posture and spine axis.
# - After calibration, the script tracks maximal right and left axial rotation
#   (thorax vs pelvis around calibrated spine axis).
#
# Usage:
#   1. Start script with patient in their best possible upright sitting posture.
#   2. Wait for "Calibration done" message.
#   3. Ask patient to rotate as far as COMFORTABLE to the right and left, a few times.
#   4. When you're satisfied with measured ROM, press Ctrl+C.
#   5. Script prints ROM and suggested thresholds for the main rehab code.
#
# You can then set in your main script:
#   AXIAL_TARGET_RIGHT_DEG = e.g. 0.6 * right_rom_deg
#   AXIAL_TARGET_LEFT_DEG  = e.g. 0.6 * left_rom_deg

import socket
import time
import sys
import numpy as np
from ahrs.filters import Madgwick
from scipy.signal import savgol_filter
from datetime import datetime

# IMU streaming server (your WiFi device)
HOST = "192.168.4.1"
PORT = 3333

HEADERS = ["imu_id","time_ms","acc_x_g","acc_y_g","acc_z_g",
           "pitch_deg","roll_deg","gyr_x_dps","gyr_y_dps","gyr_z_dps"]

# ------- Tunables -------

SAMPLE_HZ           = 20.0
CALIBRATION_SECONDS = 3.0   # personal upright calibration

# Stream channel IDs (same as your main script)
UPPER_ID  = "IMU_CH3"
PELVIS_ID = "IMU_CH0"

# Stillness gating
STILL_GYR_DPS   = 2.5
G_1G            = 1.0
ACC_STILL_TOL_G = 0.05  # ±5%

# Pairing
PAIR_MAX_AGE_S  = 0.15
BAR_CLIP_DEG    = 60.0

# Smoothing for display / peak detection
SMOOTH_WINDOW   = 5
SMOOTH_POLY     = 2

# Relative-gyro fusion settings (to be consistent with main code)
REL_LEAK_TAU_STILL_S    = 6.0
REL_MIN_DT_FRAC         = 0.4     # dt guard fraction of 1/SAMPLE_HZ
REL_MAX_INT_OFFSET_DEG  = 25.0

def beep():
    print("\a", end="", flush=True)

def color(txt, c):
    codes = {
        "red":"\033[31m", "green":"\033[32m", "yellow":"\033[33m",
        "blue":"\033[34m", "mag":"\033[35m", "cyan":"\033[36m",
        "bold":"\033[1m", "reset":"\033[0m"
    }
    return f"{codes.get(c,'')}{txt}{codes['reset']}"

def bar(value, span=60, max_abs=60.0):
    v = float(np.clip(value, -max_abs, max_abs))
    half = span//2
    pos = int(round((v/max_abs)*half)) if max_abs>0 else 0
    left = "-"*(half+min(0,pos))
    mid  = "|"
    right= "-"*(half-max(pos,0))
    return f"[{left}{mid}{right}] {v:+.1f}°"

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def euler_zyx(q):
    """Return roll, pitch, yaw (deg) from quaternion (Z-Y-X convention)."""
    w,x,y,z = q
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    sinp  = 2*(w*y - z*x)
    sinp  = np.clip(sinp, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    return roll, pitch, yaw

def quat_to_R(q):
    """Quaternion to 3x3 rotation matrix (body->world)."""
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ])

def is_still(gyr_rad_s, acc_g):
    """Stillness test: low gyro magnitude + accel magnitude ~1g."""
    gyr_dps = np.degrees(np.linalg.norm(gyr_rad_s))
    acc_mag = np.linalg.norm(acc_g)
    return (gyr_dps < STILL_GYR_DPS) and (abs(acc_mag - G_1G) <= ACC_STILL_TOL_G)

def angle_wrap_deg(a):
    """Wrap angle to [-180, 180) degrees."""
    return ((a + 180.0) % 360.0) - 180.0

def axial_angle_about_axis(q_rel, axis_world):
    """
    Approximate trunk rotation angle (deg) around given axis_world using the
    relative quaternion q_rel (upper vs pelvis).

    q_rel = [w, x, y, z] with |q_rel|=1.
    For a pure rotation about axis_world, v = axis_world * sin(theta/2).
    We project v onto axis_world to get the component of rotation around that axis.

    Returns signed angle in degrees, wrapped to [-180, 180].
    """
    axis = np.array(axis_world, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return 0.0
    axis /= n
    w, x, y, z = q_rel
    v = np.array([x, y, z])
    v_par = np.dot(v, axis)  # component along axis
    v_par = np.clip(v_par, -1.0, 1.0)
    theta = 2.0 * np.degrees(np.arcsin(v_par))  # signed
    return angle_wrap_deg(theta)

def connect_imu():
    """Connect to the IMU streaming server (over WiFi)."""
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
        gx,gy,gz = map(float, parts[7:10])
        gx,gy,gz = np.radians([gx,gy,gz])  # rad/s
        return imu_id, t_ms, np.array([ax,ay,az]), np.array([gx,gy,gz])
    except Exception:
        return None

def now_hhmmss():
    return datetime.now().strftime("%H:%M:%S")

def main():
    print(color("=== Trunk rotation ROM measurement (axial about calibrated spine axis) ===", "bold"))
    print("Instructions:")
    print(" 1) Place IMUs as in the main rehab setup.")
    print(" 2) Ask patient to sit in their best possible upright posture.")
    print(" 3) Start script and let it calibrate for ~3 seconds.")
    print(" 4) Then ask patient to rotate as far RIGHT and LEFT as comfortable a few times.")
    print(" 5) When done, press Ctrl+C to see measured ROM and suggested thresholds.\n")

    f_upper = Madgwick(beta=0.08, frequency=SAMPLE_HZ)
    f_pel   = Madgwick(beta=0.08, frequency=SAMPLE_HZ)
    q_u = np.array([1.,0.,0.,0.])
    q_p = np.array([1.,0.,0.,0.])

    # Freeze snapshots (for true stillness)
    q_u_freeze = q_u.copy()
    q_p_freeze = q_p.copy()
    frozen = False

    # Calibration state
    start_ms = None
    t0_dev   = None
    yaw_rel_bias = None
    pel_yaw_bias = None

    coll_rel = []   # axial angle during calibration
    coll_pel = []   # pelvis yaw during calibration

    gyr_u_cal = []
    gyr_p_cal = []

    # Relative posture at calibration: thorax vs pelvis
    coll_p_rel = []
    coll_r_rel = []
    p_rel_bias = 0.0
    r_rel_bias = 0.0

    # Gyro biases
    gyr_u_bias = np.zeros(3)
    gyr_p_bias = np.zeros(3)

    # Differential-gyro axial integrator (deg)
    rel_axial_int = 0.0

    # Smoothing & timing
    yaw_buf = []
    wall_prev = None

    # Paired frames
    last_u_wall = None
    last_p_wall = None
    new_u = False
    new_p = False

    # Latest raw acc/gyro for each
    acc_u = np.array([0.,0.,1.])
    gyr_u = np.zeros(3)
    acc_p = np.array([0.,0.,1.])
    gyr_p = np.zeros(3)

    # Spine axis: body-frame definition and calibrated world-frame axis
    spine_axis_body = np.array([1.0, 0.0, 0.0])  # IMU +X ≈ spine up
    spine_axis_world_calib = None                # set after calibration

    # ROM tracking
    max_right_deg = 0.0   # largest positive axial angle
    max_left_deg  = 0.0   # most negative axial angle

    def render(axial_deg, yaw_vel, right_deg, left_deg):
        bar_txt = bar(axial_deg, span=60, max_abs=BAR_CLIP_DEG)
        right = [
            f"axial {axial_deg:+.1f}°",
            f"vel {yaw_vel:+.0f}°/s",
            f"ROM_R {right_deg:+.1f}°",
            f"ROM_L {left_deg:+.1f}°",
        ]
        print((f"{bar_txt} | " + " | ".join(right)).ljust(120), end="\r", flush=True)

    def print_event(msg, tone="green"):
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(f"[{now_hhmmss()}] " + color(msg, tone))

    # ---- IMU Socket ----
    s = connect_imu()
    buffer = ""
    header_seen = False

    try:
        while True:
            data = s.recv(1024)
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
                    header_seen = True

                parsed = parse_line(line)
                if parsed is None:
                    print_event(f"UNPARSED: {line}", tone="yellow")
                    continue

                imu_id, t_ms, acc, gyr = parsed
                if start_ms is None:
                    start_ms = t_ms
                t_dev = (t_ms - start_ms)/1000.0
                if t0_dev is None:
                    t0_dev = t_dev

                # ---- Update filters ----
                if imu_id == UPPER_ID:
                    acc_u = acc.copy()
                    gyr_u = gyr.copy()
                    g_for_filter = gyr if yaw_rel_bias is None else (gyr - gyr_u_bias)
                    q_u = f_upper.updateIMU(gyr=g_for_filter, acc=acc, q=q_u)
                    last_u_wall = time.perf_counter()
                    new_u = True
                    if (t_dev - t0_dev) <= CALIBRATION_SECONDS:
                        gyr_u_cal.append(gyr)

                elif imu_id == PELVIS_ID:
                    acc_p = acc.copy()
                    gyr_p = gyr.copy()
                    g_for_filter = gyr if yaw_rel_bias is None else (gyr - gyr_p_bias)
                    q_p = f_pel.updateIMU(gyr=g_for_filter, acc=acc, q=q_p)
                    last_p_wall = time.perf_counter()
                    new_p = True
                    if (t_dev - t0_dev) <= CALIBRATION_SECONDS:
                        gyr_p_cal.append(gyr)

                else:
                    continue

                # Process only when both are fresh and recent (paired frames)
                if not (new_u and new_p):
                    continue
                noww = time.perf_counter()
                if (noww - last_u_wall) > PAIR_MAX_AGE_S or (noww - last_p_wall) > PAIR_MAX_AGE_S:
                    continue
                new_u = new_p = False

                # Rotation matrices (for gyro transforms)
                Ru = quat_to_R(q_u)
                Rp = quat_to_R(q_p)

                # ---- Stillness check and freezing (after calibration) ----
                still_u = is_still(gyr_u - (gyr_u_bias if yaw_rel_bias is not None else 0.0), acc_u)
                still_p = is_still(gyr_p - (gyr_p_bias if yaw_rel_bias is not None else 0.0), acc_p)

                if yaw_rel_bias is not None and still_u and still_p:
                    if not frozen:
                        q_u_freeze = q_u.copy()
                        q_p_freeze = q_p.copy()
                        frozen = True
                    q_u = q_u_freeze.copy()
                    q_p = q_p_freeze.copy()
                    Ru = quat_to_R(q_u)
                    Rp = quat_to_R(q_p)
                else:
                    frozen = False

                # ---- Relative orientation (thorax vs pelvis) ----
                q_rel = quat_mul(q_u, quat_conj(q_p))
                r_rel, p_rel, _ = euler_zyx(q_rel)

                # Pelvis yaw
                _, _, yaw_p = euler_zyx(q_p)

                # During calibration, use instantaneous pelvis +X
                spine_axis_world_dynamic = Rp @ spine_axis_body
                axial_geom_calib = axial_angle_about_axis(q_rel, spine_axis_world_dynamic)

                # ---- Calibration window ----
                if (t_dev - t0_dev) <= CALIBRATION_SECONDS:
                    coll_rel.append(axial_geom_calib)
                    coll_pel.append(yaw_p)
                    coll_p_rel.append(p_rel)
                    coll_r_rel.append(r_rel)

                    pct = min(1.0, (t_dev - t0_dev)/CALIBRATION_SECONDS)
                    render(0.0, 0.0, max_right_deg, max_left_deg)
                    print(f" Calibrating… {pct*100:0.0f}%".ljust(25), end="\r", flush=True)
                    wall_prev = time.perf_counter()
                    continue

                # Finalize calibration once
                if yaw_rel_bias is None:
                    yaw_rel_bias = float(np.median(coll_rel)) if coll_rel else 0.0
                    pel_yaw_bias = float(np.median(coll_pel)) if coll_pel else 0.0

                    if gyr_u_cal:
                        gyr_u_bias = np.median(np.vstack(gyr_u_cal), axis=0)
                    if gyr_p_cal:
                        gyr_p_bias = np.median(np.vstack(gyr_p_cal), axis=0)

                    if coll_p_rel:
                        p_rel_bias = float(np.median(coll_p_rel))
                    if coll_r_rel:
                        r_rel_bias = float(np.median(coll_r_rel))

                    spine_axis_world_calib = Rp @ spine_axis_body
                    n_axis = np.linalg.norm(spine_axis_world_calib)
                    if n_axis > 1e-6:
                        spine_axis_world_calib /= n_axis
                    else:
                        spine_axis_world_calib = np.array([1.0, 0.0, 0.0])

                    print_event(
                        f"Calibration done. Axial bias {yaw_rel_bias:+.1f}°, "
                        f"pelvis yaw bias {pel_yaw_bias:+.1f}°. "
                        f"Rel posture bias (roll, pitch) = {r_rel_bias:+.1f}°, {p_rel_bias:+.1f}°",
                        tone="green"
                    )
                    print_event("Now ask the patient to rotate as far RIGHT and LEFT as comfortable.", tone="yellow")
                    rel_axial_int = 0.0
                    wall_prev = time.perf_counter()
                    continue

                # ---- Normal measurement phase ----

                axis_used = spine_axis_world_calib if spine_axis_world_calib is not None else (Rp @ spine_axis_body)

                axial_geom = axial_angle_about_axis(q_rel, axis_used)

                axial_bc_raw = angle_wrap_deg(axial_geom - yaw_rel_bias)
                axial_bc     = axial_bc_raw

                # Wall-clock dt
                t_now = time.perf_counter()
                dt = 0.0 if wall_prev is None else (t_now - wall_prev)
                wall_prev = t_now

                # Differential-gyro axial rate
                gw_u = Ru @ (gyr_u - gyr_u_bias)
                gw_p = Rp @ (gyr_p - gyr_p_bias)
                rel_omega_world = gw_u - gw_p

                axis = axis_used
                norm_axis = np.linalg.norm(axis)
                if norm_axis < 1e-6:
                    axis_unit = np.array([1.0, 0.0, 0.0])
                else:
                    axis_unit = axis / norm_axis

                axial_omega_rad = np.dot(rel_omega_world, axis_unit)
                axial_omega_dps = np.degrees(axial_omega_rad)

                # Movement decision (for fusion)
                moving = (np.degrees(np.linalg.norm(gw_u)) >= STILL_GYR_DPS) \
                      or (np.degrees(np.linalg.norm(gw_p)) >= STILL_GYR_DPS)

                # Integrate axial gyro with leak
                if dt > (REL_MIN_DT_FRAC / SAMPLE_HZ):
                    if moving:
                        rel_axial_int += axial_omega_dps * dt
                    else:
                        rel_axial_int *= np.exp(-dt / REL_LEAK_TAU_STILL_S)
                        alpha = 0.35
                        rel_axial_int = (1.0 - alpha)*rel_axial_int + alpha*axial_bc

                # Prevent runaway integrator drift
                diff_int_geom = angle_wrap_deg(rel_axial_int - axial_bc)
                if abs(diff_int_geom) > REL_MAX_INT_OFFSET_DEG:
                    rel_axial_int = axial_bc + np.clip(
                        diff_int_geom,
                        -REL_MAX_INT_OFFSET_DEG,
                        REL_MAX_INT_OFFSET_DEG,
                    )
                rel_axial_int = angle_wrap_deg(rel_axial_int)

                # Complementary fusion
                if moving:
                    w_gyro = 0.3
                else:
                    w_gyro = 0.0
                axial_fused = (1.0 - w_gyro)*axial_bc + w_gyro*rel_axial_int
                axial_fused = angle_wrap_deg(axial_fused)

                # Smoothing
                yaw_buf.append(axial_fused)
                if len(yaw_buf) >= max(3, SMOOTH_WINDOW):
                    if len(yaw_buf) >= SMOOTH_WINDOW and SMOOTH_WINDOW % 2 == 1:
                        y_smooth = savgol_filter(
                            np.array(yaw_buf[-SMOOTH_WINDOW:]),
                            SMOOTH_WINDOW,
                            SMOOTH_POLY
                        )[-1]
                    else:
                        y_smooth = yaw_buf[-1]
                else:
                    y_smooth = axial_fused

                yaw_vel_dps = 0.0
                # Simple finite-difference velocity (only if we have a bit of history)
                if len(yaw_buf) > 1 and dt > (REL_MIN_DT_FRAC / SAMPLE_HZ):
                    yaw_vel_dps = (y_smooth - yaw_buf[-2]) / dt

                # --- Update ROM ---
                if y_smooth > max_right_deg:
                    max_right_deg = y_smooth
                    beep()
                    print_event(f"New RIGHT ROM peak: {max_right_deg:+.1f}°", tone="cyan")

                if y_smooth < max_left_deg:
                    max_left_deg = y_smooth
                    beep()
                    print_event(f"New LEFT ROM peak: {max_left_deg:+.1f}°", tone="cyan")

                # Live display
                render(y_smooth, yaw_vel_dps, max_right_deg, max_left_deg)

    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print("Stopping ROM measurement…")
    except (ConnectionError, OSError) as e:
        sys.stdout.write("\n")
        print(f"Connection error: {e}")
    finally:
        try:
            s.close()
        except Exception:
            pass

        # Final ROM summary
        right_rom = max(0.0, max_right_deg)
        left_rom  = abs(min(0.0, max_left_deg))

        print(color("\n=== ROM Measurement Summary ===", "bold"))
        print(f"  Right ROM (max): +{right_rom:.1f}°")
        print(f"  Left  ROM (max): -{left_rom:.1f}° (negative direction)")

        # Example: use 60% of ROM as targets for training
        frac = 0.6
        right_target = right_rom * frac
        left_target  = left_rom  * frac

        print("\nSuggested target thresholds for main rehab script (e.g. 60% of ROM):")
        print(f"  AXIAL_TARGET_RIGHT_DEG = {right_target:.1f}")
        print(f"  AXIAL_TARGET_LEFT_DEG  = {left_target:.1f}")
        print("\nYou can adjust the fraction (e.g. 0.5, 0.7) depending on how hard you want the task.")
        print("Done.")

if __name__ == "__main__":
    main()