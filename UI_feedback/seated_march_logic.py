import time
import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt
import warnings 
import sys 

# =========================
# CONFIG AND CONSTANTS
# =========================
FS = 50.0 
LOW, HIGH, ORDER = 0.25, 2.5, 4
VM_PEAK_THR_G = 0.01   # Candidate peak threshold (Ajustado)
MAX_STEP_RATE_SPM = 35   
MIN_STEP_S = 2.0         
MAX_STEP_S = 4.0         
COACH_MIN_AMP_G = 0.03   # Umbral de "buen ejercicio"
BILATERAL_FACTOR = 2.0
SINGLE_IMU_ONE_LEG = True

# --- STREAMING/BUFFER CONTROL ---
MAX_BUFFER_SIZE_SECONDS = 2.0  # Reducido para eficiencia
BATCH_WINDOW_SECONDS = 0.6     # 0.6s (30 frames), mínimo para el filtro
BATCH_WINDOW_FRAMES = int(FS * BATCH_WINDOW_SECONDS)
MAX_BUFFER_SIZE_FRAMES = int(FS * MAX_BUFFER_SIZE_SECONDS) 


# =========================
# 1. SIGNAL UTILITIES (Reduced Prints)
# =========================

def bandpass_vm(ax, ay, az, fs=FS, low=LOW, high=HIGH, order=ORDER):
    vm = np.sqrt(ax*ax + ay*ay + az*az)
    vm = vm - np.nanmean(vm)
    if len(vm) < 28:
        # Keep this for diagnosis if buffer is empty
        # print(f"[Bandpass] Buffer too short ({len(vm)} < 28). Returning raw VM.", flush=True)
        return vm
    sos = butter(order, [low/(fs/2), high/(fs/2)], btype="band", output="sos")
    vm_f = sosfiltfilt(sos, vm)
    
    # Only print VM stats if amplitude is low (helps diagnose why peaks = 0)
    if np.max(vm_f) < VM_PEAK_THR_G:
        print(f"[DEBUG:Bandpass] VM_f LOW Amplitude: max={np.max(vm_f):.4f}g (THR={VM_PEAK_THR_G}g)", flush=True)
    return vm_f

def detect_candidate_peaks(signal, fs=FS, min_height=VM_PEAK_THR_G, min_distance_s=0.1):
    min_distance = int(min_distance_s * fs)
    peaks, last_i = [], -10**9
    for i in range(1, len(signal)-1):
        if signal[i] > min_height and signal[i] > signal[i-1] and signal[i] >= signal[i+1]:
            if i - last_i >= min_distance:
                peaks.append(i); last_i = i
    
    if len(peaks) == 0:
        # Only print if no peaks are found, as this is the primary failure point
        # print(f"[DEBUG:Peaks] Candidate Peaks found: {len(peaks)}", flush=True)
        pass 
    return np.array(peaks, dtype=int)

def merge_step_peaks(peaks, signal, fs=FS, max_step_rate_spm=MAX_STEP_RATE_SPM):
    if len(peaks) == 0: return peaks
    min_step_period = 60.0 / max_step_rate_spm       
    cluster_window = int((min_step_period/2.0) * fs) 
    
    selected = []
    cluster = [peaks[0]]
    for p in peaks[1:]:
        if p - cluster[-1] <= cluster_window:
            cluster.append(p)
        else:
            best = max(cluster, key=lambda idx: signal[idx])
            selected.append(best)
            cluster = [p]
    selected.append(max(cluster, key=lambda idx: signal[idx]))
    # print(f"[DEBUG:Merge] Peaks after merging: {len(selected)}", flush=True)
    return np.array(selected, dtype=int)

def filter_steps_by_timing(peaks, fs=FS, min_step_s=MIN_STEP_S, max_step_s=MAX_STEP_S):
    if len(peaks) == 0: return peaks
    
    valid, last_t = [], None
    for idx in peaks:
        t_cur = idx / fs
        if last_t is None: 
            valid.append(idx); last_t = t_cur
        else:
            dt_s = t_cur - last_t
            if dt_s < min_step_s:
                # print(f"[DEBUG:Timing] Peak skipped (dt={dt_s:.2f}s < MIN={min_step_s}s)", flush=True)
                continue
            valid.append(idx); last_t = t_cur
    
    # print(f"[DEBUG:Timing] Final Steps: {len(valid)} (MIN_S={min_step_s}s)", flush=True)
    return np.array(valid, dtype=int)

def detect_steps(vm_f, fs=FS):
    peaks_cand = detect_candidate_peaks(vm_f, fs=fs, min_height=VM_PEAK_THR_G, min_distance_s=0.1)
    peaks_merged = merge_step_peaks(peaks_cand, vm_f, fs=fs, max_step_rate_spm=MAX_STEP_RATE_SPM)
    steps = filter_steps_by_timing(peaks_merged, fs=fs, min_step_s=MIN_STEP_S, max_step_s=MAX_STEP_S)
    return steps


def compute_coaching_flags(raw_df):
    """
    Core analysis: Runs the full detection pipeline on the buffer.
    Returns: (n_steps, last_amp, negative_feedback_bool, step_indices_in_batch)
    """
    if raw_df.empty or len(raw_df) < 28: 
        return (0, 0.0, False, np.array([]))

    ax = pd.to_numeric(raw_df["acc_x_g"],  errors="coerce").values.astype(float)
    ay = pd.to_numeric(raw_df["acc_y_g"],  errors="coerce").values.astype(float)
    az = pd.to_numeric(raw_df["acc_z_g"],  errors="coerce").values.astype(float)

    vm_f = bandpass_vm(ax, ay, az)
    step_idx = detect_steps(vm_f, fs=FS)
    n_steps = len(step_idx)

    if n_steps == 0: 
        # print(f"[DEBUG:Coaching] No steps detected in analysis window ({len(raw_df)} frames).", flush=True)
        return (0, 0.0, False, step_idx)

    latest_amp = vm_f[step_idx[-1]]
    
    amp_low = latest_amp < COACH_MIN_AMP_G
    negative_feedback = amp_low
    
    return (n_steps, latest_amp, negative_feedback, step_idx)


# =========================
# 2. EXERCISE CLASS (Count Fix and Reduced Prints)
# =========================

class SeatedMarchLogic:
    def __init__(self):
        self.last_analysis_step_count = 0 
        self.total_session_steps = 0 # <-- NUEVO: Contador global de pasos
        self.data_buffer = pd.DataFrame(columns=["time_ms", "acc_x_g", "acc_y_g", "acc_z_g"])
        self.max_buffer_size = MAX_BUFFER_SIZE_FRAMES 
        print("[Logic] Seated/Standing March logic loaded. Max buffer size:", self.max_buffer_size, "frames.", flush=True)

    def check_calmness(self, raw_data_dict):
        self.last_analysis_step_count = 0 
        self.total_session_steps = 0 # Resetear el contador global al calibrar
        self.data_buffer = self.data_buffer.iloc[0:0].copy() 
        print("[DEBUG:Calmness] Buffer and step count cleared.", flush=True)
        return True 

    def analyze_performance(self, raw_data_dict):
        
        # Accessing GLOBAL_STATE from the main controller
        try:
            current_state = sys.modules['imu_main_controller'].GLOBAL_STATE['current_exercise']
        except (KeyError, ImportError, AttributeError):
            current_state = 'SEATED_MARCH' 

        if current_state != 'SEATED_MARCH' and current_state != 'STANDING_MARCH':
            return None
        
        if 'sensor2' not in raw_data_dict:
            return None

        # 1. Update Data Buffer
        new_row = {
            "time_ms": time.time() * 1000.0,
            "acc_x_g": raw_data_dict['sensor2']['accel_x'], 
            "acc_y_g": raw_data_dict['sensor2']['accel_y'],
            "acc_z_g": raw_data_dict['sensor2']['accel_z'],
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.data_buffer = pd.concat([self.data_buffer, pd.DataFrame([new_row])], ignore_index=True)

        current_len = len(self.data_buffer)
        
        # 2. Analysis check (only when enough data is present)
        if current_len >= BATCH_WINDOW_FRAMES: 
            
            analysis_data = self.data_buffer.iloc[-MAX_BUFFER_SIZE_FRAMES:].copy()
            
            n_steps_in_analysis, latest_amp, negative_feedback, step_idx = compute_coaching_flags(analysis_data) 
            
            # 3. CRITICAL: Check for NEW Step Event
            if n_steps_in_analysis > self.last_analysis_step_count:
                
                # Update global step count
                self.total_session_steps += (n_steps_in_analysis - self.last_analysis_step_count)
                
                # --- PRINT EVENT (NEW) ---
                amp_status = "Amp Low (BAD)" if negative_feedback else "Amp OK (GOOD)"
                # Reportamos el contador global
                print(f"\n[STEP DETECTED] Total Steps: {self.total_session_steps} | Last Amp: {latest_amp:.3f}g | Status: {amp_status} (THR={COACH_MIN_AMP_G}g)", flush=True)
                # -------------------------

                # 4. Generate Feedback based on the NEW step
                feedback_to_send = negative_feedback # True for BAD, False for GOOD
                
                # 5. Buffer Management (CRITICAL: Reset counter after trimming)
                self.last_analysis_step_count = n_steps_in_analysis
                
                if len(step_idx) > 0:
                    last_step_index_in_batch = step_idx[-1]
                    self.data_buffer = self.data_buffer.iloc[last_step_index_in_batch:].reset_index(drop=True)
                    self.last_analysis_step_count = 0 # Reset for the new, smaller buffer
                    # print(f"[DEBUG:Buffer Trim] Trimmed buffer after step. New size: {len(self.data_buffer)} frames.", flush=True)
                
                return feedback_to_send
            
            # 6. Limit buffer size (if no step was detected but buffer is too large)
            if current_len > self.max_buffer_size:
                self.data_buffer = self.data_buffer.iloc[-self.max_buffer_size:].reset_index(drop=True)
                self.last_analysis_step_count = 0
                # print(f"[DEBUG:Buffer Limit] Buffer trimmed (max size reached). New size: {len(self.data_buffer)} frames.", flush=True)
            
            return None 
        else:
            # print(f"[DEBUG:Buffer] Accumulating data: {current_len}/{BATCH_WINDOW_FRAMES} frames.", flush=True)
            return None