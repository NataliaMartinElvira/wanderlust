# seated_trunk_rotation_axial_realtime_turnstate_unity_tcp_fixed.py
#
# Seated trunk rotation detection using two IMUs:
# - 3 s time-based calibration (assumed still)
# - Trunk rotation = rotation of thorax vs pelvis around a calibrated pelvis "spine axis" (IMU +X),
#   relative to that calibration posture (which may be flexed / non-ideal)
# - Robust to small mounting differences (X≈up, Y≈left, Z≈back)
# - Drift-controlled via differential gyro fusion projected onto joint axis + stillness freezing
# - Side-specific targets for right and left
# - REAL-TIME, per-turn feedback:
#   * "Rotate further to the right/left" (only while actively moving deeper into that side)
#   * "Keep hips still while turning right/left"
#   * "Keep your trunk more upright while turning right/left"
#   * "Slow down while turning right/left"
#   Feedback is tied to an active "turn" state and updated every ~0.4 s at most.
#
# UNITY TCP UI INTEGRATION:
# - Unity runs a TCP server, e.g. at 127.0.0.1:5001.
# - This script connects as a client and sends ONE-LINE EVENTS (no side info):
#       UI_CONNECTED
#       IMU_CONNECTED
#       CALIB_DONE
#       REP_REACHED
#       REP_TOO_FAST
#       REP_TOO_SLOW
#       SLOW_DOWN
#       UPRIGHT
#       HIPS_STILL
#       ROTATE_FURTHER
#
# DRIFT / FEEDBACK IMPROVEMENTS:
# - Each warning type (speed / bending / pelvis / depth) is given at most once per turn.
# - Hysteresis on bending and pelvis drift to avoid chatter.
# - "moving" depends only on gyro, so holding a rotated posture is considered rest.
# - Relative-gyro integrator is prevented from drifting far from geometric angle.
# - Fused angles are wrapped so they stay in [-180, 180]°.
#
# REP-SPEED CLASSIFICATION:
# - When a rep (left or right) hits its target, the duration from turn start is measured.
# - If duration < REP_MIN_DURATION_S  -> "rep too fast"  (REP_TOO_FAST)
# - If duration > REP_MAX_DURATION_S  -> "rep too slow"  (REP_TOO_SLOW)
# - Else treated as normal-speed rep (only REP_REACHED event).

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

# Unity UI TCP server (Unity listens here)
UI_HOST = "127.0.0.1"
UI_PORT = 5001

# ------- Tunables -------

SAMPLE_HZ               = 20.0
CALIBRATION_SECONDS     = 3.0   # assume patient sits still (their personal upright)

# Coaching targets (side-specific)
AXIAL_TARGET_RIGHT_DEG  = 10.0   # target axial rotation for RIGHT (positive)
AXIAL_TARGET_LEFT_DEG   = 10.0   # target axial rotation for LEFT  (positive magnitude)
YAW_HYST_DEG            = 5.0    # hysteresis around each target

MAX_YAW_SPEED_DPS       = 120.0  # instantaneous max speed for "Slow down"
MAX_COMP_ANGLE_DEG      = 15.0   # max deviation from calibrated relative posture
MAX_PELVIS_DRIFT_DEG    = 10.0   # max pelvis yaw drift from calibration

# Rep logic
MIN_REP_GAP_S           = 1.0
SMOOTH_WINDOW           = 5
SMOOTH_POLY             = 2

# Rep-speed classification (per completed rep)
REP_MIN_DURATION_S      = 0.5    # reps shorter than this are "too fast"
REP_MAX_DURATION_S      = 3.0    # reps longer than this are "too slow"

# Stream channel IDs
UPPER_ID  = "IMU_CH3"
PELVIS_ID = "IMU_CH0"

# Stillness gating (freeze both only when BOTH are truly still)
STILL_GYR_DPS           = 2.5
G_1G                    = 1.0
ACC_STILL_TOL_G         = 0.05  # ±5%

# Pairing + UI
PAIR_MAX_AGE_S          = 0.15
BAR_CLIP_DEG            = 60.0

# Relative-gyro fusion settings (drift control)
REL_LEAK_TAU_STILL_S    = 6.0    # leak time constant at rest (seconds)
REL_MIN_DT_FRAC         = 0.4    # dt guard fraction of 1/SAMPLE_HZ
REL_MAX_INT_OFFSET_DEG  = 25.0   # max allowed difference between integrator and geometry

# Real-time per-turn feedback settings
DIR_MIN_DEG                 = 5.0   # |axial| to consider "actively turning"
TURN_END_DEG                = 3.0   # |axial| below this => turn stopped
TURN_FEEDBACK_INTERVAL_S    = 0.4   # min time between messages during SAME turn
DEPTH_WARN_FRACTION         = 0.6   # say "rotate further" if below this fraction of target
DEPTH_WARN_MIN_TIME_S       = 0.2   # wait a short time before depth warning
MIN_DEPTH_VEL_DPS           = 3.0   # min velocity in same direction to give "rotate further"

# ---------- Utilities ----------

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

# ---------- Unity TCP integration ----------

ui_socket = None

def connect_ui_socket():
    """Try once to connect to Unity TCP server."""
    global ui_socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect((UI_HOST, UI_PORT))
        s.settimeout(None)
        ui_socket = s
        print(color(f"[UI] Connected to Unity at {UI_HOST}:{UI_PORT}", "cyan"))
        # Inform Unity that UI link is established
        send_ui_event("UI_CONNECTED")
    except Exception as e:
        print(color(f"[UI] Could not connect to Unity: {e}", "yellow"))
        ui_socket = None

def send_ui_event(event_name: str):
    """
    Send a single event line to Unity over TCP.
    Example events:
        "UI_CONNECTED"
        "IMU_CONNECTED"
        "CALIB_DONE"
        "REP_REACHED"
        "REP_TOO_FAST"
        "REP_TOO_SLOW"
        "SLOW_DOWN"
        "UPRIGHT"
        "HIPS_STILL"
        "ROTATE_FURTHER"
    """
    global ui_socket
    if ui_socket is None:
        return
    try:
        msg = event_name.strip().upper() + "\n"
        ui_socket.sendall(msg.encode("utf-8"))
    except Exception as e:
        print(color(f"[UI] Send failed, closing socket: {e}", "yellow"))
        try:
            ui_socket.close()
        except Exception:
            pass
        ui_socket = None

# ---------- Main ----------

def main():
    print(color("Seated trunk rotation – axial about calibrated pelvis spine axis (real-time per-turn feedback)", "bold"))
    print("Sit still for ~3 s at start (posture + gyro-bias calibration).")
    print(
        f"Targets: Right axial ≥ {AXIAL_TARGET_RIGHT_DEG:.0f}°, "
        f"Left axial ≤ -{AXIAL_TARGET_LEFT_DEG:.0f}°, "
        f"speed ≤ {MAX_YAW_SPEED_DPS:.0f}°/s, "
        f"comp ≤ {MAX_COMP_ANGLE_DEG:.0f}°, pelvis drift ≤ {MAX_PELVIS_DRIFT_DEG:.0f}°.\n"
    )

    # Try to connect to Unity once at startup
    connect_ui_socket()

    f_upper = Madgwick(beta=0.08, frequency=SAMPLE_HZ)
    f_pel   = Madgwick(beta=0.08, frequency=SAMPLE_HZ)
    q_u = np.array([1.,0.,0.,0.])
    q_p = np.array([1.,0.,0.,0.])

    # Freeze snapshots (for true stillness)
    q_u_freeze = q_u.copy()
    q_p_freeze = q_p.copy()
    frozen = False

    # Rep zones
    zone_pos = False
    zone_neg = False

    # Calibration state
    start_ms = None
    t0_dev   = None
    yaw_rel_bias = None
    pel_yaw_bias = None

    coll_rel = []   # axial angle (about spine axis) during calibration
    coll_pel = []   # pelvis yaw during calibration

    gyr_u_cal = []
    gyr_p_cal = []

    # Relative posture at calibration: thorax vs pelvis
    coll_p_rel = []   # pitch_rel
    coll_r_rel = []   # roll_rel
    p_rel_bias = 0.0
    r_rel_bias = 0.0

    # Gyro biases
    gyr_u_bias = np.zeros(3)
    gyr_p_bias = np.zeros(3)

    # Differential-gyro axial integrator (deg)
    rel_axial_int = 0.0

    # Rep tracking
    last_yaw = None
    last_peak_t = -1e9
    peaks_L = []
    peaks_R = []

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

    # Turn state for real-time feedback
    turn_state_dir = 0          # 0 = none, +1 right, -1 left
    turn_state_start_t = 0.0
    turn_state_last_fb_t = -1e9

    # Track which warnings have already been given in the current turn
    warned_speed_this_turn  = False
    warned_bend_this_turn   = False
    warned_pelvis_this_turn = False
    warned_depth_this_turn  = False

    # Hysteretic states for bending and pelvis drift
    bending_state = False
    pelvis_state  = False

    # Spine axis: body-frame definition and calibrated world-frame axis
    spine_axis_body = np.array([1.0, 0.0, 0.0])  # IMU +X ≈ spine up
    spine_axis_world_calib = None                # set after calibration

    def print_event(msg, tone="green"):
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(f"[{now_hhmmss()}] " + color(msg, tone))

    def render(y_smooth, yaw_vel, pel_dev):
        # Compact status line for live view only
        bar_txt = bar(y_smooth, span=60, max_abs=BAR_CLIP_DEG)
        right = [
            f"axial {y_smooth:+.1f}°",
            f"vel {yaw_vel:+.0f}°/s",
            f"pelvis {pel_dev:+.1f}°",
        ]
        print((f"{bar_txt} | " + " | ".join(right)).ljust(110), end="\r", flush=True)

    # ---- IMU Socket ----
    s = connect_imu()
    # Tell Unity that IMU/WiFi connection is established
    send_ui_event("IMU_CONNECTED")

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
                r_rel, p_rel, _ = euler_zyx(q_rel)  # relative roll/pitch for bending check

                # Pelvis yaw (for drift/hip movement check)
                _, _, yaw_p = euler_zyx(q_p)

                # During calibration, use instantaneous pelvis +X axis
                spine_axis_world_dynamic = Rp @ spine_axis_body

                # Geometric axial angle about spine axis (for calibration collection)
                axial_geom_calib = axial_angle_about_axis(q_rel, spine_axis_world_dynamic)

                # ---- Calibration window ----
                if (t_dev - t0_dev) <= CALIBRATION_SECONDS:
                    # Collect axial angle and pelvis yaw baselines
                    coll_rel.append(axial_geom_calib)
                    coll_pel.append(yaw_p)

                    # Collect relative posture baselines (thorax vs pelvis)
                    coll_p_rel.append(p_rel)
                    coll_r_rel.append(r_rel)

                    pct = min(1.0, (t_dev - t0_dev)/CALIBRATION_SECONDS)
                    render(0.0, 0.0, 0.0)
                    print(f" Calibrating… {pct*100:0.0f}%".ljust(20), end="\r", flush=True)
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

                    # Relative posture baselines (so rotations are w.r.t. this posture)
                    if coll_p_rel:
                        p_rel_bias = float(np.median(coll_p_rel))
                    if coll_r_rel:
                        r_rel_bias = float(np.median(coll_r_rel))

                    # Calibrated spine axis in world coordinates (personal "upright")
                    spine_axis_world_calib = Rp @ spine_axis_body
                    n_axis = np.linalg.norm(spine_axis_world_calib)
                    if n_axis > 1e-6:
                        spine_axis_world_calib /= n_axis
                    else:
                        spine_axis_world_calib = np.array([1.0, 0.0, 0.0])

                    print_event(
                        f"Calibration done. "
                        f"Axial bias {yaw_rel_bias:+.1f}°, pelvis yaw bias {pel_yaw_bias:+.1f}°. "
                        f"Gyro biases z (upper, pelvis) = "
                        f"{np.degrees(gyr_u_bias[2]):+.3f}°/s, {np.degrees(gyr_p_bias[2]):+.3f}°/s. "
                        f"Rel posture bias (roll, pitch) = {r_rel_bias:+.1f}°, {p_rel_bias:+.1f}°",
                        tone="green"
                    )

                    # Inform Unity that calibration is done
                    send_ui_event("CALIB_DONE")

                    rel_axial_int = 0.0
                    last_yaw = None
                    wall_prev = time.perf_counter()
                    continue  # next loop iteration with calibrated biases

                # ---- Normal measurement phase ----

                # Use calibrated spine axis if available; fall back to current if not
                axis_used = spine_axis_world_calib if spine_axis_world_calib is not None else (Rp @ spine_axis_body)

                # Geometric axial angle about calibrated spine axis
                axial_geom = axial_angle_about_axis(q_rel, axis_used)

                # Bias-corrected axial angle and pelvis deviation
                # NOTE: positive = right, negative = left
                axial_bc_raw = angle_wrap_deg(axial_geom - yaw_rel_bias)
                axial_bc     = axial_bc_raw

                y_pel_dev = angle_wrap_deg(yaw_p - pel_yaw_bias)       # pelvis yaw vs calibration

                # Wall-clock dt
                t_now = time.perf_counter()
                dt = 0.0 if wall_prev is None else (t_now - wall_prev)
                wall_prev = t_now

                # Differential-gyro axial rate: project relative angular velocity onto calibrated spine axis
                gw_u = Ru @ (gyr_u - gyr_u_bias)   # rad/s, world frame
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

                # Movement decision: ONLY gyro-based, not angle-based
                moving = (np.degrees(np.linalg.norm(gw_u)) >= STILL_GYR_DPS) \
                      or (np.degrees(np.linalg.norm(gw_p)) >= STILL_GYR_DPS)

                # Integrate axial gyro with leak + glue to geometric axial angle at rest
                if dt > (REL_MIN_DT_FRAC / SAMPLE_HZ):
                    if moving:
                        # During movement, follow gyro changes (short-term)
                        rel_axial_int += axial_omega_dps * dt
                    else:
                        # At rest, gently decay & re-align to geometry to limit drift
                        rel_axial_int *= np.exp(-dt / REL_LEAK_TAU_STILL_S)
                        alpha = 0.35   # geometry weight at rest
                        rel_axial_int = (1.0 - alpha)*rel_axial_int + alpha*axial_bc

                # ---- Prevent runaway integrator drift ----
                diff_int_geom = angle_wrap_deg(rel_axial_int - axial_bc)
                if abs(diff_int_geom) > REL_MAX_INT_OFFSET_DEG:
                    rel_axial_int = axial_bc + np.clip(
                        diff_int_geom,
                        -REL_MAX_INT_OFFSET_DEG,
                        REL_MAX_INT_OFFSET_DEG,
                    )
                rel_axial_int = angle_wrap_deg(rel_axial_int)

                # Complementary fusion: geometry-dominant, gyro-assisted when moving
                if moving:
                    w_gyro = 0.3
                else:
                    w_gyro = 0.0
                axial_fused = (1.0 - w_gyro)*axial_bc + w_gyro*rel_axial_int

                # Wrap fused angle as well
                axial_fused = angle_wrap_deg(axial_fused)

                # Smoothing & axial velocity (for display and rep detection)
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
                if dt > (REL_MIN_DT_FRAC / SAMPLE_HZ) and last_yaw is not None:
                    yaw_vel_dps = (y_smooth - last_yaw) / dt
                last_yaw = y_smooth

                # ---- Quality signals ----
                p_rel_dev = p_rel - p_rel_bias
                r_rel_dev = r_rel - r_rel_bias

                raw_bending_now = (
                    abs(p_rel_dev) > MAX_COMP_ANGLE_DEG or
                    abs(r_rel_dev) > MAX_COMP_ANGLE_DEG
                )
                raw_pelvis_violation_now = abs(y_pel_dev) > MAX_PELVIS_DRIFT_DEG
                too_fast_now = abs(yaw_vel_dps) > MAX_YAW_SPEED_DPS

                # Simple hysteresis of ~3 degrees to avoid chatter
                BEND_OUT_DEG   = max(0.0, MAX_COMP_ANGLE_DEG - 3.0)
                PELVIS_OUT_DEG = max(0.0, MAX_PELVIS_DRIFT_DEG - 3.0)

                # Bending hysteresis
                if bending_state:
                    if (abs(p_rel_dev) <= BEND_OUT_DEG and
                        abs(r_rel_dev) <= BEND_OUT_DEG):
                        bending_state = False
                else:
                    if raw_bending_now:
                        bending_state = True

                # Pelvis drift hysteresis
                if pelvis_state:
                    if abs(y_pel_dev) <= PELVIS_OUT_DEG:
                        pelvis_state = False
                else:
                    if raw_pelvis_violation_now:
                        pelvis_state = True

                bending_now = bending_state
                pelvis_violation_now = pelvis_state

                # ---- REAL-TIME per-turn feedback state machine ----
                abs_axial = abs(y_smooth)  # use smoothed angle for direction & thresholds

                if abs_axial > DIR_MIN_DEG and moving:
                    # Direction from smoothed angle (more stable)
                    current_dir = 1 if y_smooth >= 0 else -1

                    # Start new turn or switch direction
                    if turn_state_dir == 0 or current_dir != turn_state_dir:
                        turn_state_dir = current_dir
                        turn_state_start_t = t_dev
                        turn_state_last_fb_t = -1e9  # allow immediate feedback

                        # Reset per-turn warning flags
                        warned_speed_this_turn  = False
                        warned_bend_this_turn   = False
                        warned_pelvis_this_turn = False
                        warned_depth_this_turn  = False

                    # Decide if it's time to say something
                    time_in_turn = t_dev - turn_state_start_t
                    time_since_fb = t_dev - turn_state_last_fb_t
                    if time_since_fb >= TURN_FEEDBACK_INTERVAL_S:
                        dir_sign = turn_state_dir
                        dir_str = "right" if dir_sign > 0 else "left"
                        target_for_dir = AXIAL_TARGET_RIGHT_DEG if dir_sign > 0 else AXIAL_TARGET_LEFT_DEG

                        # Is the motion currently going deeper into this side?
                        same_direction_motion = (yaw_vel_dps * dir_sign) > MIN_DEPTH_VEL_DPS

                        msgs = []

                        # Speed warning
                        if too_fast_now and not warned_speed_this_turn:
                            msgs.append(f"Slow down while turning {dir_str}.")
                            warned_speed_this_turn = True
                            send_ui_event("SLOW_DOWN")

                        # Bending warning (extra forward/side lean beyond calibration)
                        if bending_now and not warned_bend_this_turn:
                            msgs.append(f"Keep your trunk more upright while turning {dir_str}.")
                            warned_bend_this_turn = True
                            send_ui_event("UPRIGHT")

                        # Pelvis / hip movement warning
                        if pelvis_violation_now and not warned_pelvis_this_turn:
                            msgs.append(f"Keep your hips still while turning {dir_str}.")
                            warned_pelvis_this_turn = True
                            send_ui_event("HIPS_STILL")

                        # Depth / "rotate further" feedback
                        if (
                            time_in_turn > DEPTH_WARN_MIN_TIME_S and
                            same_direction_motion and
                            abs_axial < DEPTH_WARN_FRACTION * target_for_dir and
                            not warned_depth_this_turn
                        ):
                            msgs.append(f"Rotate further to the {dir_str}.")
                            warned_depth_this_turn = True
                            send_ui_event("ROTATE_FURTHER")

                        if msgs:
                            beep()
                            print_event(" ".join(msgs), tone="yellow")
                            turn_state_last_fb_t = t_dev

                else:
                    # No longer clearly turning -> reset turn state
                    if abs_axial < TURN_END_DEG or not moving:
                        turn_state_dir = 0
                        turn_state_start_t = t_dev
                        turn_state_last_fb_t = -1e9
                        warned_speed_this_turn  = False
                        warned_bend_this_turn   = False
                        warned_pelvis_this_turn = False
                        warned_depth_this_turn  = False

                # ---- Rep detection (side-specific thresholds with hysteresis) ----
                up_th    = AXIAL_TARGET_RIGHT_DEG        # right target
                down_th  = -AXIAL_TARGET_LEFT_DEG        # left target (negative)
                exit_pos = up_th   - YAW_HYST_DEG        # right hysteresis exit
                exit_neg = down_th + YAW_HYST_DEG        # left hysteresis exit (less negative)

                # Right side (positive axial)
                if (not zone_pos) and (y_smooth >= up_th) and ((t_dev - last_peak_t) >= MIN_REP_GAP_S):
                    zone_pos = True
                    zone_neg = False
                    peaks_R.append((t_dev, y_smooth))
                    last_peak_t = t_dev
                    beep()
                    print_event(f"✓ Target reached (Right) at {y_smooth:+.1f}°", tone="green")
                    send_ui_event("REP_REACHED")

                    # ---- Rep-speed classification for this rep ----
                    rep_duration = t_dev - turn_state_start_t
                    if rep_duration < REP_MIN_DURATION_S:
                        print_event(
                            f"Rep too fast (Right): {rep_duration:.2f}s < {REP_MIN_DURATION_S:.2f}s",
                            tone="yellow"
                        )
                        send_ui_event("REP_TOO_FAST")
                    elif rep_duration > REP_MAX_DURATION_S:
                        print_event(
                            f"Rep too slow (Right): {rep_duration:.2f}s > {REP_MAX_DURATION_S:.2f}s",
                            tone="yellow"
                        )
                        send_ui_event("REP_TOO_SLOW")

                # Left side (negative axial)
                if (not zone_neg) and (y_smooth <= down_th) and ((t_dev - last_peak_t) >= MIN_REP_GAP_S):
                    zone_neg = True
                    zone_pos = False
                    peaks_L.append((t_dev, y_smooth))
                    last_peak_t = t_dev
                    beep()
                    print_event(f"✓ Target reached (Left) at {y_smooth:+.1f}°", tone="green")
                    send_ui_event("REP_REACHED")

                    # ---- Rep-speed classification for this rep ----
                    rep_duration = t_dev - turn_state_start_t
                    if rep_duration < REP_MIN_DURATION_S:
                        print_event(
                            f"Rep too fast (Left): {rep_duration:.2f}s < {REP_MIN_DURATION_S:.2f}s",
                            tone="yellow"
                        )
                        send_ui_event("REP_TOO_FAST")
                    elif rep_duration > REP_MAX_DURATION_S:
                        print_event(
                            f"Rep too slow (Left): {rep_duration:.2f}s > {REP_MAX_DURATION_S:.2f}s",
                            tone="yellow"
                        )
                        send_ui_event("REP_TOO_SLOW")

                # Hysteresis exit
                if zone_pos and y_smooth < exit_pos:
                    zone_pos = False
                if zone_neg and y_smooth > exit_neg:
                    zone_neg = False

                # Live status line (compact)
                render(y_smooth, yaw_vel_dps, y_pel_dev)

    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print("Stopping…")
    except (ConnectionError, OSError) as e:
        sys.stdout.write("\n")
        print(f"Connection error: {e}")
    finally:
        try:
            s.close()
        except Exception:
            pass

        global ui_socket
        if ui_socket is not None:
            try:
                ui_socket.close()
            except Exception:
                pass

        nL, nR = len(peaks_L), len(peaks_R)
        print(color(f"\nSession summary: peaks L={nL}, R={nR}", "bold"))
        if peaks_L:
            peaks_L_values = [v for _, v in peaks_L]
            print(f"  Left mean peak:  {np.mean(peaks_L_values):.1f}°")
        if peaks_R:
            peaks_R_values = [v for _, v in peaks_R]
            print(f"  Right mean peak: {np.mean(peaks_R_values):.1f}°")
        if peaks_L and peaks_R:
            print(f"  Asymmetry (R-L): {np.mean([v for _, v in peaks_R]) - np.mean([v for _, v in peaks_L]):+.1f}°")
        print("Done.")

if __name__ == "__main__":
    main()
