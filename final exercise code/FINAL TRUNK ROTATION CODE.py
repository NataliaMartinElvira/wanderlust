# code to detect and evaluate trunk rotations based on the data from 2 IMUs; sends feedback to terminal and Unity UI via TCP

# IMPORTS
import msvcrt
import serial 
import time
import sys
import numpy as np
from ahrs.filters import Madgwick
from scipy.signal import savgol_filter
from datetime import datetime
import socket

# =========================
# CONFIGURATION
# =========================
SERIAL_PORT = 'COM8'      # <--- THIS DEPENDS ON THE PORT OF THE LAPTOP WHERE THE RECEIVER IS PLUGGED IN
BAUD_RATE = 115200

# Packet Config for V2_ACCEL
LEN_V2 = 17 

# Unity UI TCP server
UI_HOST = "127.0.0.1"
UI_PORT = 5001

# ------- Tunables -------

SAMPLE_HZ               = 20.0
CALIBRATION_SECONDS     = 3.0   # assume patient sits still (their personal upright)

# Coaching targets
AXIAL_TARGET_RIGHT_DEG  = 7.0   # based on preliminary testing with patient, target for right rotation
AXIAL_TARGET_LEFT_DEG   = 7.0   # based on preliminary testing with patient, target for left rotation
YAW_HYST_DEG            = 5.0   

MAX_YAW_SPEED_DPS       = 120.0 
MAX_COMP_ANGLE_DEG      = 20.0  # Increased from 15.0 to 20.0 based on testing; allows more natural trunk bending
MAX_PELVIS_DRIFT_DEG    = 20.0  # Increased from 15.0 to 20.0 based on testing; allows more natural pelvis movement (hips)

# Rep logic
MIN_REP_GAP_S           = 1.0
SMOOTH_WINDOW           = 5
SMOOTH_POLY             = 2
REP_MIN_DURATION_S      = 0.5   
REP_MAX_DURATION_S      = 3.0   

# Neutral Re-Arm Logic: when system is allowed to count new rep (Prevents swing-through false reps)
NEUTRAL_BAND_DEG        = 3.0   # Must return within +/- this angle to re-arm
NEUTRAL_DWELL_S         = 0.3   # Must stay in neutral for this long to re-arm

# Stillness gating
STILL_GYR_DPS           = 1.5
G_1G                    = 1.0
ACC_STILL_TOL_G         = 0.05  

# Pairing + UI
PAIR_MAX_AGE_S          = 0.15
BAR_CLIP_DEG            = 60.0

# Relative-gyro fusion settings (Axial Drift Control)
REL_LEAK_TAU_STILL_S    = 6.0    
REL_MIN_DT_FRAC         = 0.4    
REL_MAX_INT_OFFSET_DEG  = 25.0   

# NEW: Pelvis Drift Control (Leaky Integrator)
PELVIS_LEAK_TAU_S       = 4.0    

# Real-time per-turn feedback settings
DIR_MIN_DEG                 = 3.1   
TURN_END_DEG                = 3.0   
TURN_FEEDBACK_INTERVAL_S    = 0.4   
DEPTH_WARN_FRACTION         = 0.9   
DEPTH_WARN_MIN_TIME_S       = 0.7 
MIN_DEPTH_VEL_DPS           = 1.0   

# Auto-tare settings
AUTO_TARE_MIN_REPS         = 1      
AUTO_TARE_NO_REP_S         = 4.0    # Reduced from 8.0 to recover quickly if center drifts
AXIAL_AUTO_TARE_MIN_DEG    = 8.0    
AXIAL_AUTO_TARE_MAX_DEG    = 45.0   
AUTO_TARE_MAX_VEL_DPS      = 5.0    


# ---------- Utilities ----------

def beep():
    print("\a", end="", flush=True)

def color(txt, c): # for coloured terminal output
    codes = {
        "red":"\033[31m", "green":"\033[32m", "yellow":"\033[33m",
        "blue":"\033[34m", "mag":"\033[35m", "cyan":"\033[36m",
        "bold":"\033[1m", "reset":"\033[0m"
    }
    return f"{codes.get(c,'')}{txt}{codes['reset']}"

def bar(value, span=60, max_abs=60.0): # horizontal bar in terminal output to visualize trunk angle
    v = float(np.clip(value, -max_abs, max_abs))
    half = span//2
    pos = int(round((v/max_abs)*half)) if max_abs>0 else 0
    left = "-"*(half+min(0,pos))
    mid  = "|"
    right= "-"*(half-max(pos,0))
    return f"[{left}{mid}{right}] {v:+.1f}°"

def quat_conj(q): # quaternion conjugate
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

def euler_zyx(q): # Euler conversion
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

def is_still(gyr_rad_s, acc_g): # Checks if sensor is “still”: low gyro magnitude and accel magnitude close to 1g. --> important for freezing drift and detecting movement
    """Stillness test: low gyro magnitude + accel magnitude ~1g."""
    gyr_dps = np.degrees(np.linalg.norm(gyr_rad_s))
    acc_mag = np.linalg.norm(acc_g)
    return (gyr_dps < STILL_GYR_DPS) and (abs(acc_mag - G_1G) <= ACC_STILL_TOL_G)

def angle_wrap_deg(a):
    """Wrap angle to [-180, 180) degrees to avoid unbounded drift."""
    return ((a + 180.0) % 360.0) - 180.0

def axial_angle_about_axis(q_rel, axis_world): # axial rotation about the spine axis
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

def now_hhmmss(): # time 
    return datetime.now().strftime("%H:%M:%S")

# ---------- Unity TCP integration ----------
''' UI sends: 
CALIB_DONE
IMU_CONNECTED
REP_REACHED
SLOW_DOWN
UPRIGHT
ROTATE_FURTHER
HIPS_STILL'''

ui_socket = None

def connect_ui_socket():
    global ui_socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect((UI_HOST, UI_PORT))
        s.settimeout(None)
        ui_socket = s
        print(color(f"[UI] Connected to Unity at {UI_HOST}:{UI_PORT}", "cyan"))
        send_ui_event("UI_CONNECTED")
    except Exception as e:
        print(color(f"[UI] Could not connect to Unity: {e}", "yellow"))
        ui_socket = None

def send_ui_event(event_name: str):
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
    print(color("Seated trunk rotation – V2 DUAL IMU (Serial) + Unity", "bold"))
    print("Sit still for ~3 s at start (posture + gyro-bias calibration).") # start with calibration prompt
    print(
        f"Targets: Right axial ≥ {AXIAL_TARGET_RIGHT_DEG:.0f}°, "
        f"Left axial ≤ -{AXIAL_TARGET_LEFT_DEG:.0f}°, "
        f"speed ≤ {MAX_YAW_SPEED_DPS:.0f}°/s, "
        f"comp ≤ {MAX_COMP_ANGLE_DEG:.0f}°, pelvis drift ≤ {MAX_PELVIS_DRIFT_DEG:.0f}°.\n"
    )

    connect_ui_socket()

    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE}...")
    try:
        s_serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Connected to Serial.\n")
        send_ui_event("IMU_CONNECTED")
    except Exception as e:
        print(color(f"Serial Error: {e}", "red"))
        return

    # Two Madgwick filters: one for upper body, one for pelvis; and current quaternions for each
    f_upper = Madgwick(beta=0.08, frequency=SAMPLE_HZ)
    f_pel   = Madgwick(beta=0.08, frequency=SAMPLE_HZ)
    q_u = np.array([1.,0.,0.,0.])
    q_p = np.array([1.,0.,0.,0.])

    #Frozen quaternions (used when user is extremely still)
    q_u_freeze = q_u.copy()
    q_p_freeze = q_p.copy()
    frozen = False

    # Rep Arming Logic (for counting valid reps)
    zone_pos = False # inside right target zone
    zone_neg = False # inside left target zone
    
    
    reps_armed = True   # Start armed (assuming starting at neutral), can currently count a new rep
    neutral_entry_time = None # when we entered neutral band

    start_ms = None
    t0_dev   = None
    yaw_rel_bias = None # (computed after calibration)
    pel_yaw_bias = None

    coll_rel = []  # axial angles during calibration
    coll_pel = []  # pelvis yaw during calibration

    gyr_u_cal = [] # raw gyro samples (for bias)
    gyr_p_cal = []

    coll_p_rel = [] # pitch of relative orientation (for bending bias)
    coll_r_rel = [] 
    p_rel_bias = 0.0
    r_rel_bias = 0.0

    gyr_u_bias = np.zeros(3) # gyro bias after calibration
    gyr_p_bias = np.zeros(3)

    # Differential-gyro axial integrator (deg)
    rel_axial_int = 0.0
    
    # Pelvis Leaky Integrator
    pelvis_yaw_int = 0.0

    # Rep tracking
    last_yaw = None
    last_peak_t = -1e9
    peaks_L = []
    peaks_R = []

    # Smoothing buffer for yaw angle
    yaw_buf = []
    wall_prev = None

    # Current raw sensor arrays, initialized
    acc_u = np.array([0.,0.,1.])
    gyr_u = np.zeros(3)
    acc_p = np.array([0.,0.,1.])
    gyr_p = np.zeros(3)

    turn_state_dir = 0   # 0=not turning, +1=right, -1=left
    turn_state_start_t = 0.0
    turn_state_last_fb_t = -1e9

    # warning flags per turn
    warned_speed_this_turn  = False
    warned_bend_this_turn   = False
    warned_pelvis_this_turn = False
    warned_depth_this_turn  = False

    bending_state = False
    pelvis_state  = False

    # Spine axis definition
    spine_axis_body = np.array([1.0, 0.0, 0.0]) 
    spine_axis_world_calib = None 

    def print_event(msg, tone="green"): # print timestamped colored log messages
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(f"[{now_hhmmss()}] " + color(msg, tone))

    # Render function to build single status line with ASCII bar, status (ARMED / WAIT), axial angle, angular velocity, pelvis yaw dev, pitch/roll dev (bending)
    def render(y_smooth, yaw_vel, pel_dev, p_dev, r_dev):
        status = "ARMED" if reps_armed else "WAIT "
        bar_txt = bar(y_smooth, span=55, max_abs=BAR_CLIP_DEG)
        right = [
            f"[{status}]",
            f"axial {y_smooth:+.1f}°",
            f"vel {yaw_vel:+.0f}°/s",
            f"pelvis {pel_dev:+.1f}°",
            f"bend P{p_dev:+.0f}° R{r_dev:+.0f}°" # Added Pitch and Roll feedback
        ]
        # Joined string
        line = (f"{bar_txt} | " + " | ".join(right))
        # Pad to clear prev line (120 chars usually safe for full width)
        print(line.ljust(120), end="\r", flush=True)

    ### Main loop (tries to read IMU data from serial, process, and provide feedback)
    try:
        while True:
            if s_serial.in_waiting > 0:
                line = s_serial.readline().decode("utf-8", errors="ignore").strip()
                
                if not line:
                    continue

                if line.startswith("V2_ACCEL:"): # Only processes lines starting with that
                    clean_line = line.replace("V2_ACCEL:", "")
                    parts = clean_line.split(',') # splits by comma to get values
                    
                    if len(parts) == LEN_V2:
                        try:
                            t_ms = int(float(parts[0]))
                            
                            # pelvis and upper IMU accelerations (g) and gyros (deg/s → rad/s)
                            ax_p, ay_p, az_p = float(parts[1]), float(parts[2]), float(parts[3])
                            gx_p, gy_p, gz_p = float(parts[6]), float(parts[7]), float(parts[8])
                            acc_p = np.array([ax_p, ay_p, az_p])
                            gyr_p = np.radians([gx_p, gy_p, gz_p])

                            ax_u, ay_u, az_u = float(parts[9]), float(parts[10]), float(parts[11])
                            gx_u, gy_u, gz_u = float(parts[14]), float(parts[15]), float(parts[16])
                            acc_u = np.array([ax_u, ay_u, az_u])
                            gyr_u = np.radians([gx_u, gy_u, gz_u])
                            
                            # Time in “device seconds since start”
                            if start_ms is None:
                                start_ms = t_ms
                            t_dev = (t_ms - start_ms)/1000.0
                            if t0_dev is None:
                                t0_dev = t_dev

                            # ==========================================
                            # LOGIC PIPELINE
                            # ==========================================
                            
                            # Madgwick filter!: B efore calibration finished, uses raw gyros. After calibration, subtracts estimated gyro bias.
                            g_u_filt = gyr_u if yaw_rel_bias is None else (gyr_u - gyr_u_bias)
                            q_u = f_upper.updateIMU(gyr=g_u_filt, acc=acc_u, q=q_u)
                            
                            g_p_filt = gyr_p if yaw_rel_bias is None else (gyr_p - gyr_p_bias)
                            q_p = f_pel.updateIMU(gyr=g_p_filt, acc=acc_p, q=q_p)

                            # Collect Gyro Calibration Data
                            if (t_dev - t0_dev) <= CALIBRATION_SECONDS:
                                gyr_u_cal.append(gyr_u)
                                gyr_p_cal.append(gyr_p)

                            # Convert to rotation matrices
                            Ru = quat_to_R(q_u)
                            Rp = quat_to_R(q_p)

                            # Stillness check
                            still_u = is_still(gyr_u - (gyr_u_bias if yaw_rel_bias is not None else 0.0), acc_u)
                            still_p = is_still(gyr_p - (gyr_p_bias if yaw_rel_bias is not None else 0.0), acc_p)

                            # Freeze logic: Once calibrated and both IMUs are very still, orientation is frozen to last steady value to fight tiny jitter
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

                            # Relative orientation
                            q_rel = quat_mul(q_u, quat_conj(q_p))
                            r_rel, p_rel, _ = euler_zyx(q_rel) # used for trunk bending (pitch/roll)
                            # Note: We still calculate absolute Yaw for calibration refs, but not for realtime feedback
                            _, _, yaw_p = euler_zyx(q_p)

                            # Dynamic spine axis during calibration: For calibration phase, uses pelvis orientation at that instant to define trunk axis in world frame and compute axial rotation
                            spine_axis_world_dynamic = Rp @ spine_axis_body
                            axial_geom_calib = axial_angle_about_axis(q_rel, spine_axis_world_dynamic)

                            # ---- CALIBRATION PHASE ----
                            if (t_dev - t0_dev) <= CALIBRATION_SECONDS:
                                coll_rel.append(axial_geom_calib)
                                coll_pel.append(yaw_p)
                                coll_p_rel.append(p_rel)
                                coll_r_rel.append(r_rel)

                                pct = min(1.0, (t_dev - t0_dev)/CALIBRATION_SECONDS)
                                render(0.0, 0.0, 0.0, 0.0, 0.0) # Pass placeholders
                                print(f" Calibrating… {pct*100:0.0f}%".ljust(20), end="\r", flush=True)
                                wall_prev = time.perf_counter()
                                continue 

                            # ---- FINALIZE CALIBRATION ----
                            if yaw_rel_bias is None:
                                yaw_rel_bias = float(np.median(coll_rel)) if coll_rel else 0.0
                                pel_yaw_bias = float(np.median(coll_pel)) if coll_pel else 0.0

                                if gyr_u_cal:
                                    gyr_u_bias = np.median(np.vstack(gyr_u_cal), axis=0)
                                if gyr_p_cal:
                                    gyr_p_bias = np.median(np.vstack(gyr_p_cal), axis=0)

                                if coll_p_rel: p_rel_bias = float(np.median(coll_p_rel))
                                if coll_r_rel: r_rel_bias = float(np.median(coll_r_rel))

                                spine_axis_world_calib = Rp @ spine_axis_body
                                n_axis = np.linalg.norm(spine_axis_world_calib)
                                if n_axis > 1e-6:
                                    spine_axis_world_calib /= n_axis
                                else:
                                    spine_axis_world_calib = np.array([1.0, 0.0, 0.0])
                                
                                print_event(
                                    f"Calibration done. Bias u={np.degrees(gyr_u_bias[2]):.3f}, p={np.degrees(gyr_p_bias[2]):.3f}",
                                    tone="green"
                                )
                                send_ui_event("CALIB_DONE")

                                rel_axial_int = 0.0
                                pelvis_yaw_int = 0.0 # Reset integrator
                                last_yaw = None
                                wall_prev = time.perf_counter()
                                continue

                            # ---- MEASUREMENT PHASE (CALIBRATION IS DONE) ----
                            axis_used = spine_axis_world_calib if spine_axis_world_calib is not None else (Rp @ spine_axis_body)
                            axial_geom = axial_angle_about_axis(q_rel, axis_used) # absolute axial rotation about calibrated spine axis
                            
                            axial_bc = angle_wrap_deg(axial_geom - yaw_rel_bias) # angle relative to neutral (0 = neutral)
                            
                            # Wall clock DT
                            t_now = time.perf_counter()
                            dt = 0.0 if wall_prev is None else (t_now - wall_prev)
                            wall_prev = t_now

                            # Differential gyro rate
                            gw_u = Ru @ (gyr_u - gyr_u_bias)
                            gw_p = Rp @ (gyr_p - gyr_p_bias)
                            rel_omega_world = gw_u - gw_p # relative angular velocity
                            
                            axis_unit = axis_used / (np.linalg.norm(axis_used) + 1e-9)
                            axial_omega_dps = np.degrees(np.dot(rel_omega_world, axis_unit)) # Projects onto spine axis → axial rotational velocity (deg/s)

                            # Movement check: “Moving” if either IMU’s world-frame gyro magnitude exceeds stillness threshold
                            moving = (np.degrees(np.linalg.norm(gw_u)) >= STILL_GYR_DPS) \
                                   or (np.degrees(np.linalg.norm(gw_p)) >= STILL_GYR_DPS)

                            # -----------------------------------------------------
                            # PELVIS LEAKY INTEGRATOR: Exponential leak slowly drives it back to zero when not moving (prevents unbounded drift)
                            # -----------------------------------------------------
                            pelvis_rate_z = np.degrees(gw_p[2]) # Z-axis rotation rate
                            
                            if dt > (REL_MIN_DT_FRAC / SAMPLE_HZ):
                                pelvis_yaw_int += pelvis_rate_z * dt
                                if PELVIS_LEAK_TAU_S > 0:
                                    leak_factor = np.exp(-dt / PELVIS_LEAK_TAU_S) # Exponential leak
                                    pelvis_yaw_int *= leak_factor
                                
                            y_pel_dev = pelvis_yaw_int

                            # Axial Integration
                            ''' When moving: integrate relative axial rate. When still: slowly decay integrator and blend back to geometric angle'''
                            if dt > (REL_MIN_DT_FRAC / SAMPLE_HZ):
                                if moving:
                                    rel_axial_int += axial_omega_dps * dt
                                else:
                                    rel_axial_int *= np.exp(-dt / REL_LEAK_TAU_STILL_S)
                                    alpha = 0.35
                                    rel_axial_int = (1.0 - alpha)*rel_axial_int + alpha*axial_bc

                            # Drift Clamp
                            diff_int_geom = angle_wrap_deg(rel_axial_int - axial_bc)
                            if abs(diff_int_geom) > REL_MAX_INT_OFFSET_DEG:
                                rel_axial_int = axial_bc + np.clip(diff_int_geom, -REL_MAX_INT_OFFSET_DEG, REL_MAX_INT_OFFSET_DEG)
                            rel_axial_int = angle_wrap_deg(rel_axial_int)

                            # Fusion
                            w_gyro = 0.3 if moving else 0.0
                            axial_fused = angle_wrap_deg((1.0 - w_gyro)*axial_bc + w_gyro*rel_axial_int)

                            # Smoothing: Savitzky–Golay 
                            yaw_buf.append(axial_fused)
                            if len(yaw_buf) >= max(3, SMOOTH_WINDOW):
                                if len(yaw_buf) >= SMOOTH_WINDOW and SMOOTH_WINDOW % 2 == 1:
                                    y_smooth = savgol_filter(np.array(yaw_buf[-SMOOTH_WINDOW:]), SMOOTH_WINDOW, SMOOTH_POLY)[-1] # Savitzky–Golay filter over last SMOOTH_WINDOW samples (odd length) to smooth axial angle
                                else:
                                    y_smooth = yaw_buf[-1]
                            else:
                                y_smooth = axial_fused

                            yaw_vel_dps = 0.0
                            if dt > (REL_MIN_DT_FRAC / SAMPLE_HZ) and last_yaw is not None:
                                yaw_vel_dps = (y_smooth - last_yaw) / dt
                            last_yaw = y_smooth

                            # ---- Quality Checks ----
                            # Compute deviations from neutral
                            p_rel_dev = p_rel - p_rel_bias
                            r_rel_dev = r_rel - r_rel_bias
                            
                            raw_bending_now = abs(p_rel_dev) > MAX_COMP_ANGLE_DEG # or abs(r_rel_dev) > MAX_COMP_ANGLE_DEG)
                            raw_pelvis_violation_now = abs(y_pel_dev) > MAX_PELVIS_DRIFT_DEG
                            too_fast_now = abs(yaw_vel_dps) > MAX_YAW_SPEED_DPS

                            # Hysteresis
                            BEND_OUT_DEG   = max(0.0, MAX_COMP_ANGLE_DEG - 3.0)
                            PELVIS_OUT_DEG = max(0.0, MAX_PELVIS_DRIFT_DEG - 3.0)

                            if bending_state:
                                if abs(p_rel_dev) <= BEND_OUT_DEG: #and abs(r_rel_dev) <= BEND_OUT_DEG): 
                                    bending_state = False
                            else:
                                if raw_bending_now: bending_state = True

                            if pelvis_state:
                                if abs(y_pel_dev) <= PELVIS_OUT_DEG: pelvis_state = False
                            else:
                                if raw_pelvis_violation_now: pelvis_state = True

                            # results 
                            bending_now = bending_state
                            pelvis_violation_now = pelvis_state
                            
                            # ---- NEUTRAL DWELL LOGIC (RE-ARM) ----
                            # You must be stable in neutral for NEUTRAL_DWELL_S to allow a new rep. 
                            if abs(y_smooth) <= NEUTRAL_BAND_DEG:
                                if neutral_entry_time is None:
                                    neutral_entry_time = t_dev
                                elif (t_dev - neutral_entry_time) >= NEUTRAL_DWELL_S:
                                    if not reps_armed:
                                        reps_armed = True

                            else:
                                neutral_entry_time = None

                            # ---- AUTO-TARE LOGIC: automatic recentering to wipe out previous drift and noise ----
                            total_reps = len(peaks_L) + len(peaks_R)
                            if (yaw_rel_bias is not None) and (total_reps >= AUTO_TARE_MIN_REPS):
                                time_since_last_rep = t_dev - last_peak_t
                                axial_offset = abs(y_smooth)
                                axial_speed  = abs(yaw_vel_dps)
                                
                                need_unlock = (not reps_armed) and (axial_speed < 2.0) and (time_since_last_rep > AUTO_TARE_NO_REP_S)
                                classic_drift = (time_since_last_rep > AUTO_TARE_NO_REP_S * 2.0) and \
                                                (AXIAL_AUTO_TARE_MIN_DEG <= axial_offset <= AXIAL_AUTO_TARE_MAX_DEG) and \
                                                (axial_speed < AUTO_TARE_MAX_VEL_DPS)

                                if need_unlock or classic_drift:
                                    yaw_rel_bias = axial_geom
                                    rel_axial_int = 0.0
                                    pelvis_yaw_int = 0.0 
                                    # Update bending biases (Pitch/Roll) to eliminate drift
                                    p_rel_bias = p_rel
                                    r_rel_bias = r_rel
                                    
                                    yaw_buf = [0.0] * SMOOTH_WINDOW
                                    last_peak_t = t_dev
                                    
                                    reps_armed = True # Force arm
                                    
                                    print_event(f"[AUTO] Re-centered (Unlock/Drift).", tone="cyan")
                                    beep()

                            # ---- REAL-TIME FEEDBACK LOGIC ----
                            # 1. Detect if currently turning
                            abs_axial = abs(y_smooth)
                            if abs_axial > DIR_MIN_DEG and moving:
                                current_dir = 1 if y_smooth >= 0 else -1
                                
                                if turn_state_dir == 0 or current_dir != turn_state_dir:
                                    turn_state_dir = current_dir
                                    turn_state_start_t = t_dev
                                    turn_state_last_fb_t = -1e9
                                    warned_speed_this_turn = False
                                    warned_bend_this_turn = False
                                    warned_pelvis_this_turn = False
                                    warned_depth_this_turn = False

                                # 2. Periodic feedback during turn
                                time_in_turn = t_dev - turn_state_start_t
                                time_since_fb = t_dev - turn_state_last_fb_t
                                
                                if time_since_fb >= TURN_FEEDBACK_INTERVAL_S:
                                    dir_sign = turn_state_dir
                                    dir_str = "right" if dir_sign > 0 else "left"
                                    target_for_dir = AXIAL_TARGET_RIGHT_DEG if dir_sign > 0 else AXIAL_TARGET_LEFT_DEG
                                    same_direction_motion = (yaw_vel_dps * dir_sign) > 0.0 
                                    
                                    msgs = []
                                    if too_fast_now and not warned_speed_this_turn:
                                        msgs.append(f"Slow down while turning {dir_str}.")
                                        warned_speed_this_turn = True
                                        send_ui_event("SLOW_DOWN")
                                    
                                    if bending_now and not warned_bend_this_turn:
                                        msgs.append(f"Keep your trunk more upright while turning {dir_str}.")
                                        warned_bend_this_turn = True
                                        send_ui_event("UPRIGHT")

                                    if pelvis_violation_now and not warned_pelvis_this_turn:
                                        msgs.append(f"Keep your hips still while turning {dir_str}.")
                                        warned_pelvis_this_turn = True
                                        send_ui_event("HIPS_STILL")

                                    if (time_in_turn > DEPTH_WARN_MIN_TIME_S and same_direction_motion and 
                                        abs_axial < target_for_dir and not warned_depth_this_turn):
                                        msgs.append(f"Rotate further to the {dir_str}.")
                                        warned_depth_this_turn = True
                                        send_ui_event("ROTATE_FURTHER")

                                    if msgs:
                                        beep()
                                        print_event(" ".join(msgs), tone="yellow")
                                        turn_state_last_fb_t = t_dev
                            # If not turning / turn ended
                            else:
                                if abs_axial < TURN_END_DEG:
                                    turn_state_dir = 0
                                    turn_state_start_t = t_dev
                                    turn_state_last_fb_t = -1e9
                                    warned_speed_this_turn = False
                                    warned_bend_this_turn = False
                                    warned_pelvis_this_turn = False
                                    warned_depth_this_turn = False

                            # ---- REP DETECTION ----
                            # Thresholds:
                            up_th    = AXIAL_TARGET_RIGHT_DEG
                            down_th  = -AXIAL_TARGET_LEFT_DEG
                            
                            # Right Rep
                            if reps_armed and (not zone_pos) and (y_smooth >= up_th) and ((t_dev - last_peak_t) >= MIN_REP_GAP_S):
                                zone_pos = True
                                zone_neg = False
                                reps_armed = False # DISARM until return to neutral
                                
                                peaks_R.append((t_dev, y_smooth))
                                last_peak_t = t_dev
                                beep()
                                print_event(f"✓ Target reached (Right) at {y_smooth:+.1f}°", tone="green")
                                send_ui_event("REP_REACHED")
                                
                                rep_duration = t_dev - turn_state_start_t
                                if rep_duration < REP_MIN_DURATION_S:
                                    print_event(f"Rep too fast (Right): {rep_duration:.2f}s", tone="yellow")
                                    send_ui_event("REP_TOO_FAST")
                                elif rep_duration > REP_MAX_DURATION_S:
                                    print_event(f"Rep too slow (Right): {rep_duration:.2f}s", tone="yellow")
                                    send_ui_event("REP_TOO_SLOW")

                            # Left Rep
                            if reps_armed and (not zone_neg) and (y_smooth <= down_th) and ((t_dev - last_peak_t) >= MIN_REP_GAP_S):
                                zone_neg = True
                                zone_pos = False
                                reps_armed = False # DISARM until return to neutral

                                peaks_L.append((t_dev, y_smooth))
                                last_peak_t = t_dev
                                beep()
                                print_event(f"✓ Target reached (Left) at {y_smooth:+.1f}°", tone="green")
                                send_ui_event("REP_REACHED")

                                rep_duration = t_dev - turn_state_start_t
                                if rep_duration < REP_MIN_DURATION_S:
                                    print_event(f"Rep too fast (Left): {rep_duration:.2f}s", tone="yellow")
                                    send_ui_event("REP_TOO_FAST")
                                elif rep_duration > REP_MAX_DURATION_S:
                                    print_event(f"Rep too slow (Left): {rep_duration:.2f}s", tone="yellow")
                                    send_ui_event("REP_TOO_SLOW")

                            # Hysteresis reset: Ensures you must move back some degrees out of the zone before another rep is possible
                            exit_pos = up_th - YAW_HYST_DEG
                            exit_neg = down_th + YAW_HYST_DEG
                            if zone_pos and y_smooth < exit_pos: zone_pos = False
                            if zone_neg and y_smooth > exit_neg: zone_neg = False

                            # UPDATED RENDER CALL: update status line
                            render(y_smooth, yaw_vel_dps, y_pel_dev, p_rel_dev, r_rel_dev)
                            
                            # MANUAL Re-Center (Tare): Manual re-center with keyboard 'z' key
                            if msvcrt.kbhit():
                                key = msvcrt.getch().decode('utf-8').lower()
                                if key == 'z':
                                    yaw_rel_bias = axial_geom
                                    rel_axial_int = 0.0 
                                    pelvis_yaw_int = 0.0 # Reset pelvis integrator
                                    # Update bending biases (Pitch/Roll) on manual tare
                                    p_rel_bias = p_rel
                                    r_rel_bias = r_rel
                                    
                                    yaw_buf = [0.0] * SMOOTH_WINDOW
                                    reps_armed = True
                                    print_event(f"!!! RE-CENTERED !!! New Bias: {yaw_rel_bias:.1f}", tone="cyan")
                                    beep()

                        except ValueError:
                            pass

# Shutdowns and exception errors 
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print("Stopping…")
    except Exception as e:
        sys.stdout.write("\n")
        print(f"Error: {e}")
    finally:
        try: s_serial.close() # close serial port
        except: pass
        if ui_socket is not None:
            try: ui_socket.close()
            except: pass

        nL, nR = len(peaks_L), len(peaks_R) # total reps left/right
        print(color(f"\nSession summary: peaks L={nL}, R={nR}", "bold"))

# run main
if __name__ == "__main__":
    main()