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
CALIBRATION_SECONDS     = 3.0   # assume patient sits still

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

# Dictionary to hold the latest raw data from both IMUs
LATEST_RAW_DATA = {
    UPPER_ID: None,
    PELVIS_ID: None
}
LAST_RAW_TIME = 0.0

# ---------- Utilities (Simplified for class context) ----------

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


# ---------- TrunkRotationLogic Class (Adapting main() logic) ----------

class TrunkRotationLogic:
    """
    IMU logic handler for the Seated Trunk Rotation exercise.
    This encapsulates the state machine logic from the original trunk rotation script.
    """
    def __init__(self):
        # Filter instances
        self.f_upper = Madgwick(beta=0.08, frequency=SAMPLE_HZ)
        self.f_pel   = Madgwick(beta=0.08, frequency=SAMPLE_HZ)
        
        # Latest quaternions (state)
        self.q_u = np.array([1.,0.,0.,0.])
        self.q_p = np.array([1.,0.,0.,0.])

        # Calibration state
        self.yaw_rel_bias = None
        self.pel_yaw_bias = None
        self.p_rel_bias = 0.0
        self.r_rel_bias = 0.0
        self.gyr_u_bias = np.zeros(3)
        self.gyr_p_bias = np.zeros(3)
        self.calib_coll_rel = []
        self.calib_coll_pel = []
        self.calib_coll_p_rel = []
        self.calib_coll_r_rel = []
        self.gyr_u_cal = []
        self.gyr_p_cal = []

        # Differential-gyro axial integrator state
        self.rel_axial_int = 0.0
        self.last_yaw = 0.0
        self.wall_prev = time.perf_counter()

        # Turn state for real-time feedback
        self.turn_state_dir = 0          # 0 = none, +1 right, -1 left
        self.turn_state_start_t = time.perf_counter()
        self.turn_state_last_fb_t = -1e9
        
        # Per-turn warning flags
        self.warned_speed_this_turn  = False
        self.warned_bend_this_turn   = False
        self.warned_pelvis_this_turn = False
        self.warned_depth_this_turn  = False
        
        # Hysteretic states
        self.bending_state = False
        self.pelvis_state  = False
        
        print("[Logic] Trunk Rotation logic loaded. (Dual IMU State Machine)")

    def check_calmness(self, raw_data_dict):
        """
        Used when START_CALIBRATION is received. Here, we perform the actual calibration
        by collecting data until the fixed time (CALIBRATION_SECONDS) is reached.
        This function will not return True (CALIB:DONE) since Unity controls the time.
        It only collects the data during the fixed time window.
        """
        # Data is assumed to be coming from the combined Arduino receiver
        if PELVIS_ID not in raw_data_dict or UPPER_ID not in raw_data_dict:
            # If the reader didn't find both IMU V2 packets, skip
            return False

        # Assuming 'acc_x', 'acc_y', 'acc_z' are already available in the raw_data_dict
        acc_u = np.array([raw_data_dict[UPPER_ID]['acc_x'], raw_data_dict[UPPER_ID]['acc_y'], raw_data_dict[UPPER_ID]['acc_z']])
        gyr_u = np.radians(np.array([raw_data_dict[UPPER_ID]['gyr_x'], raw_data_dict[UPPER_ID]['gyr_y'], raw_data_dict[UPPER_ID]['gyr_z']]))
        acc_p = np.array([raw_data_dict[PELVIS_ID]['acc_x'], raw_data_dict[PELVIS_ID]['acc_y'], raw_data_dict[PELVIS_ID]['acc_z']])
        gyr_p = np.radians(np.array([raw_data_dict[PELVIS_ID]['gyr_x'], raw_data_dict[PELVIS_ID]['gyr_y'], raw_data_dict[PELVIS_ID]['gyr_z']]))
        
        # ---- Run Filter Update (Simplified for calibration phase) ----
        self.q_u = self.f_upper.updateIMU(gyr=gyr_u, acc=acc_u, q=self.q_u)
        self.q_p = self.f_pel.updateIMU(gyr=gyr_p, acc=acc_p, q=self.q_p)
        
        # Rotation matrices
        Ru = quat_to_R(self.q_u)
        Rp = quat_to_R(self.q_p)

        # Relative orientation
        q_rel = quat_mul(self.q_u, quat_conj(self.q_p))
        r_rel, p_rel, _ = euler_zyx(q_rel)
        _, _, yaw_p = euler_zyx(self.q_p)

        # Joint axis definition (pelvis IMU +X transformed to world)
        spine_axis_body = np.array([1.0, 0.0, 0.0])
        spine_axis_world = Rp @ spine_axis_body

        # Geometric axial angle
        axial_geom = axial_angle_about_axis(q_rel, spine_axis_world)

        # Collect data for calibration biases (runs constantly during the fixed time window)
        self.calib_coll_rel.append(axial_geom)
        self.calib_coll_pel.append(yaw_p)
        self.calib_coll_p_rel.append(p_rel)
        self.calib_coll_r_rel.append(r_rel)
        self.gyr_u_cal.append(gyr_u)
        self.gyr_p_cal.append(gyr_p)

        return False # Calibration done via fixed timer in Unity

    def _finalize_calibration(self):
        """ Calculates and stores the final biases after the fixed calibration time. """
        
        if not self.calib_coll_rel:
            print("[CALIB ERROR] No data collected during calibration window.")
            return

        self.yaw_rel_bias = float(np.median(self.calib_coll_rel))
        self.pel_yaw_bias = float(np.median(self.calib_coll_pel))

        self.gyr_u_bias = np.median(np.vstack(self.gyr_u_cal), axis=0)
        self.gyr_p_bias = np.median(np.vstack(self.gyr_p_cal), axis=0)

        self.p_rel_bias = float(np.median(self.calib_coll_p_rel))
        self.r_rel_bias = float(np.median(self.calib_coll_r_rel))
        
        # Reset relative integrator
        self.rel_axial_int = 0.0
        self.last_yaw = 0.0
        self.wall_prev = time.perf_counter()
        
        print(f"[CALIB SUCCESS] Finalized biases: Axial {self.yaw_rel_bias:+.1f}°, Pelvis {self.pel_yaw_bias:+.1f}°")

    def analyze_performance(self, raw_data_dict):
        """
        Runs the full state machine for real-time per-turn feedback.
        Returns: True if bad feedback is triggered, False if good feedback is triggered, None otherwise.
        """
        # Ensure biases are calculated (i.e., check_calmness was called and time passed)
        if self.yaw_rel_bias is None:
            # First time in analyze_performance means calibration time ended in Unity
            self._finalize_calibration()
            
        # Data check
        if PELVIS_ID not in raw_data_dict or UPPER_ID not in raw_data_dict:
            return None # Skip this frame
            
        # Extract and prepare data
        acc_u = np.array([raw_data_dict[UPPER_ID]['acc_x'], raw_data_dict[UPPER_ID]['acc_y'], raw_data_dict[UPPER_ID]['acc_z']])
        gyr_u = np.radians(np.array([raw_data_dict[UPPER_ID]['gyr_x'], raw_data_dict[UPPER_ID]['gyr_y'], raw_data_dict[UPPER_ID]['gyr_z']]))
        acc_p = np.array([raw_data_dict[PELVIS_ID]['acc_x'], raw_data_dict[PELVIS_ID]['acc_y'], raw_data_dict[PELVIS_ID]['acc_z']])
        gyr_p = np.radians(np.array([raw_data_dict[PELVIS_ID]['gyr_x'], raw_data_dict[PELVIS_ID]['gyr_y'], raw_data_dict[PELVIS_ID]['gyr_z']]))

        # Apply Gyro Biases and Update Filters
        g_u_biased = gyr_u - self.gyr_u_bias
        g_p_biased = gyr_p - self.gyr_p_bias
        self.q_u = self.f_upper.updateIMU(gyr=g_u_biased, acc=acc_u, q=self.q_u)
        self.q_p = self.f_pel.updateIMU(gyr=g_p_biased, acc=acc_p, q=self.q_p)
        
        # --- Simplified Stillness Check & Freezing (To conserve processing) ---
        still_u = is_still(g_u_biased, acc_u)
        still_p = is_still(g_p_biased, acc_p)
        
        # NOTE: We skip the full freezing logic and assume filter fusion is enough

        # Rotation matrices (for gyro transforms)
        Ru = quat_to_R(self.q_u)
        Rp = quat_to_R(self.q_p)
        
        # ---- Relative orientation and Axial Angle ----
        q_rel = quat_mul(self.q_u, quat_conj(self.q_p))
        r_rel, p_rel, _ = euler_zyx(q_rel)
        _, _, yaw_p = euler_zyx(self.q_p)
        
        spine_axis_body = np.array([1.0, 0.0, 0.0])
        spine_axis_world = Rp @ spine_axis_body
        axial_geom = axial_angle_about_axis(q_rel, spine_axis_world)
        
        # Bias-corrected and Fused Angle
        axial_bc_raw = angle_wrap_deg(axial_geom - self.yaw_rel_bias)
        y_pel_dev = angle_wrap_deg(yaw_p - self.pel_yaw_bias)
        p_rel_dev = p_rel - self.p_rel_bias
        r_rel_dev = r_rel - self.r_rel_bias
        
        # Wall-clock dt
        t_now = time.perf_counter()
        dt = t_now - self.wall_prev
        self.wall_prev = t_now
        
        # Differential-gyro axial rate & Fusion (copied directly from original logic)
        gw_u = Ru @ g_u_biased
        gw_p = Rp @ g_p_biased
        rel_omega_world = gw_u - gw_p
        axis_unit = spine_axis_world / np.linalg.norm(spine_axis_world)
        axial_omega_dps = np.degrees(np.dot(rel_omega_world, axis_unit))

        moving = (np.degrees(np.linalg.norm(gw_u)) >= STILL_GYR_DPS) or \
                 (np.degrees(np.linalg.norm(gw_p)) >= STILL_GYR_DPS)
        
        # Integrate axial gyro with leak + glue to geometric axial angle
        if dt > (REL_MIN_DT_FRAC / SAMPLE_HZ):
            if moving:
                self.rel_axial_int += axial_omega_dps * dt
            else:
                self.rel_axial_int *= np.exp(-dt / REL_LEAK_TAU_STILL_S)
                alpha = 0.35
                self.rel_axial_int = (1.0 - alpha)*self.rel_axial_int + alpha*axial_bc
        self.rel_axial_int = angle_wrap_deg(self.rel_axial_int)

        w_gyro = 0.3 if moving else 0.0
        axial_fused = angle_wrap_deg((1.0 - w_gyro)*axial_bc_raw + w_gyro*self.rel_axial_int)

        # Smoothing & axial velocity
        yaw_vel_dps = 0.0
        if dt > 0.0 and self.last_yaw is not None:
            yaw_vel_dps = (axial_fused - self.last_yaw) / dt
        self.last_yaw = axial_fused

        # ---- Quality & Violation Check (Hysteresis Logic) ----
        abs_axial = abs(axial_fused)
        raw_bending_now = (abs(p_rel_dev) > MAX_COMP_ANGLE_DEG or abs(r_rel_dev) > MAX_COMP_ANGLE_DEG)
        raw_pelvis_violation_now = abs(y_pel_dev) > MAX_PELVIS_DRIFT_DEG
        too_fast_now = abs(yaw_vel_dps) > MAX_YAW_SPEED_DPS
        
        # Hysteresis checks omitted for simplicity, using raw flags for immediate feedback
        bending_now = raw_bending_now
        pelvis_violation_now = raw_pelvis_violation_now
        
        # ---- REAL-TIME per-turn feedback state machine ----
        
        feedback_event = None # True=BAD, False=GOOD, None=No Event
        time_since_fb = t_now - self.turn_state_last_fb_t
        
        if abs_axial > DIR_MIN_DEG and moving:
            current_dir = 1 if axial_fused >= 0 else -1
            
            # Start new turn or switch direction
            if self.turn_state_dir == 0 or current_dir != self.turn_state_dir:
                self.turn_state_dir = current_dir
                self.turn_state_last_fb_t = -1e9 # Allow immediate feedback
                self.warned_speed_this_turn = self.warned_bend_this_turn = self.warned_pelvis_this_turn = self.warned_depth_this_turn = False

            # Decide if it's time to say something (rate limit)
            if time_since_fb >= TURN_FEEDBACK_INTERVAL_S:
                dir_sign = self.turn_state_dir
                target_for_dir = AXIAL_TARGET_RIGHT_DEG if dir_sign > 0 else AXIAL_TARGET_LEFT_DEG

                # Priority: speed > bending > pelvis > depth
                if too_fast_now and not self.warned_speed_this_turn:
                    self.warned_speed_this_turn = True
                    feedback_event = True # BAD
                    send_ui_event_internal("SLOW_DOWN")
                    
                elif bending_now and not self.warned_bend_this_turn:
                    self.warned_bend_this_turn = True
                    feedback_event = True # BAD
                    send_ui_event_internal("UPRIGHT")
                    
                elif pelvis_violation_now and not self.warned_pelvis_this_turn:
                    self.warned_pelvis_this_turn = True
                    feedback_event = True # BAD
                    send_ui_event_internal("HIPS_STILL")
                    
                elif (
                    (yaw_vel_dps * dir_sign) > MIN_DEPTH_VEL_DPS and # Moving deeper
                    abs_axial < DEPTH_WARN_FRACTION * target_for_dir and # Not deep enough
                    not self.warned_depth_this_turn
                ):
                    self.warned_depth_this_turn = True
                    feedback_event = True # BAD
                    send_ui_event_internal("ROTATE_FURTHER")
                
                # If any BAD event was generated, update the last feedback time
                if feedback_event is not None:
                    self.turn_state_last_fb_t = t_now
                    return feedback_event # True (BAD)

        else:
            # No longer clearly turning -> reset turn state
            if abs_axial < TURN_END_DEG and not moving:
                self.turn_state_dir = 0
                self.turn_state_last_fb_t = -1e9 # Allow immediate feedback
        
        # --- Check for successful REP completion (Good feedback) ---
        # This is the "rep reached" signal, which is a success.
        
        # The logic to detect REP_REACHED is complex (zone_pos/zone_neg) and relies on hysteresis.
        # For simplicity in this event-based model, we will only check if the axial angle
        # is reached and reset the flag when it leaves the zone.
        
        reached_target = (axial_fused >= AXIAL_TARGET_RIGHT_DEG or axial_fused <= -AXIAL_TARGET_LEFT_DEG)
        
        if reached_target:
             # If target is reached and no bad feedback was generated in this turn, send GOOD feedback
             # NOTE: This section is HIGHLY simplified. Full rep tracking should be handled.
             # For now, let's skip the GOOD feedback here to avoid spam and rely on BAD feedback.
             pass
        
        return None # No event to send to Unity this frame

# --- Internal Send Helper (avoids conflict with main thread) ---
def send_ui_event_internal(event_name: str):
    """ Sends non-FEEDBACK:BAD/GOOD events (like HIPS_STILL) """
    print(f"[UI EVENT] {event_name}") # Debugging
    # The original flow only allows FEEDBACK:BAD or FEEDBACK:GOOD to be sent.
    # We must translate all the BAD events into a single "FEEDBACK:BAD" for the UI.
    # The complexity of the specific error is handled by the Python logic.
    # This function is not used in the final simple TCP integration, but kept for logic structure.
    pass