import serial
import time
import datetime as dt
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt

# =========================
# CONFIG
# =========================
SERIAL_PORT = 'COM6'      # <--- VERIFY YOUR PORT
BAUD_RATE = 115200

# PARSING CONFIG
LEN_V1 = 9 

SHEET_RAW = "IMU"
SHEET_REPS = "reps"
SHEET_SUM  = "summary"

HEADERS = [
    "time_ms", "acc_x_g", "acc_y_g", "acc_z_g",
    "pitch_deg", "roll_deg", "gyr_x_dps", "gyr_y_dps", "gyr_z_dps"
]

SAVE_EVERY_SECONDS = 2
SAVE_EVERY_ROWS = 200

# Sampling / detection params
FS = 50.0
LOW, HIGH, ORDER = 0.25, 2.5, 4

# --- MODIFIED THRESHOLD ---
VM_PEAK_THR_G = 0.0005   # <--- Set to 0.0005 as requested

# Single-IMU-on-one-leg
SINGLE_IMU_ONE_LEG = True
BILATERAL_FACTOR = 2.0

# === Coaching thresholds ===
COACH_MIN_AMP_G = 0.03    
# Note: SPM thresholds exist but won't block the step count anymore
COACH_MIN_SPM   = 8.0     
COACH_MAX_SPM   = 20.0    


def now_ts():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================
# SIGNAL UTILITIES
# =========================
def bandpass_vm(ax, ay, az, fs=FS, low=LOW, high=HIGH, order=ORDER):
    vm = np.sqrt(ax*ax + ay*ay + az*az)
    vm = vm - np.nanmean(vm)
    if len(vm) < 28: 
        return vm
    sos = butter(order, [low/(fs/2), high/(fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, vm)


def detect_candidate_peaks(signal, fs=FS, min_height=VM_PEAK_THR_G, min_distance_s=0.2):
    # Increased min_distance slightly to avoid double counting vibration, 
    # but NOT filtering for rhythm.
    min_distance = int(min_distance_s * fs)
    peaks, last_i = [], -10**9
    for i in range(1, len(signal)-1):
        if signal[i] > min_height and signal[i] > signal[i-1] and signal[i] >= signal[i+1]:
            if i - last_i >= min_distance:
                peaks.append(i); last_i = i
    return np.array(peaks, dtype=int)


def merge_step_peaks(peaks, signal, fs=FS):
    # Simplified merge logic (debounce)
    if len(peaks) == 0:
        return peaks
    cluster_window = int(0.3 * fs) 

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
    return np.array(selected, dtype=int)


def detect_steps(vm_f, fs=FS):
    # 1. Detect Peaks with new threshold
    peaks_cand = detect_candidate_peaks(vm_f, fs=fs, min_height=VM_PEAK_THR_G)
    
    # 2. Merge doubles
    peaks_merged = merge_step_peaks(peaks_cand, vm_f, fs=fs)
    
    # 3. NO TIMING FILTER
    # We return all peaks found, regardless of how slow/fast they were
    return peaks_merged


def periodicity_score(signal, fs=FS, min_p=0.4, max_p=2.0):
    x = signal - np.mean(signal)
    if len(x) < 5:
        return 0.0
    ac = np.correlate(x, x, mode="full")
    ac = ac[ac.size//2:]
    ac = ac / (ac[0] + 1e-12)
    Lmin = int(min_p*fs); Lmax = min(len(ac)-1, int(max_p*fs))
    if Lmax <= Lmin:
        return 0.0
    return float(np.max(ac[Lmin:Lmax]))


# =========================
# METRICS & REPS TABLE
# =========================
def compute_metrics(raw_df):
    if raw_df.empty:
        reps_df = pd.DataFrame(columns=[
            "t_ms","amp_g","spm_est","amp_low","too_slow","too_fast","good_rep"
        ])
        metrics = {
            "n_reps": 0, "cadence_spm": 0, "mean_amp_g": 0, "sd_amp_g": 0,
            "ipi_mean_s": 0, "ipi_sd_s": 0, "periodicity": 0
        }
        return reps_df, metrics

    t  = pd.to_numeric(raw_df["time_ms"],  errors="coerce").values.astype(float)
    ax = pd.to_numeric(raw_df["acc_x_g"],  errors="coerce").values.astype(float)
    ay = pd.to_numeric(raw_df["acc_y_g"],  errors="coerce").values.astype(float)
    az = pd.to_numeric(raw_df["acc_z_g"],  errors="coerce").values.astype(float)

    vm_f = bandpass_vm(ax, ay, az)
    step_idx = detect_steps(vm_f, fs=FS)

    rep_times = t[step_idx] if len(step_idx) else np.array([], dtype=float)
    amp       = vm_f[step_idx] if len(step_idx) else np.array([], dtype=float)
    n_steps   = len(rep_times)

    # --- DIFFERENCE OF TIME CALCULATION ---
    # We still calculate this for information, but we won't use it to block the step
    spm_est = np.full(n_steps, np.nan, dtype=float)
    if n_steps > 1:
        # Here is the difference of time logic:
        ipi_s = np.diff(rep_times) / 1000.0 
        with np.errstate(divide='ignore'):
            spm_est[1:] = 60.0 / np.clip(ipi_s, 1e-6, None)

    if SINGLE_IMU_ONE_LEG:
        spm_est = spm_est * BILATERAL_FACTOR

    if n_steps > 0:
        amp_low  = amp < COACH_MIN_AMP_G
        
        # We calculate these just for the Excel record
        valid_spm = np.isfinite(spm_est)
        too_slow = np.zeros(n_steps, dtype=bool)
        too_fast = np.zeros(n_steps, dtype=bool)
        
        too_slow[valid_spm] = spm_est[valid_spm] < COACH_MIN_SPM
        too_fast[valid_spm] = spm_est[valid_spm] > COACH_MAX_SPM
    else:
        amp_low = np.array([], dtype=bool)
        too_slow = np.array([], dtype=bool)
        too_fast = np.array([], dtype=bool)

    # --- MODIFIED EVALUATION ---
    # A rep is good if Amplitude is enough. We IGNORE timing (too_slow/too_fast).
    good_rep = (~amp_low)

    reps_df = pd.DataFrame({
        "t_ms":     rep_times,
        "amp_g":    amp,
        "spm_est":  spm_est,
        "amp_low":  amp_low,
        "too_slow": too_slow,
        "too_fast": too_fast,
        "good_rep": good_rep,
    })

    duration_min = (t[-1] - t[0]) / 60000.0 if len(t) > 1 else 0.0
    factor = (BILATERAL_FACTOR if SINGLE_IMU_ONE_LEG else 1.0)
    cadence = (n_steps * factor) / duration_min if duration_min > 0 else 0.0
    ipi = np.diff(rep_times) / 1000.0 if n_steps > 1 else np.array([], dtype=float)

    metrics = {
        "n_reps": int(n_steps),
        "cadence_spm": float(cadence),
        "mean_amp_g": float(np.nanmean(amp)) if n_steps else 0.0,
        "sd_amp_g": float(np.nanstd(amp)) if n_steps else 0.0,
        "ipi_mean_s": float(np.nanmean(ipi)) if ipi.size else 0.0,
        "ipi_sd_s": float(np.nanstd(ipi)) if ipi.size else 0.0,
        "periodicity": float(periodicity_score(vm_f))
    }
    return reps_df, metrics


def save_excel(out_path, raw_df):
    reps_df, m = compute_metrics(raw_df)
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as w:
        raw_df.to_excel(w, index=False, sheet_name=SHEET_RAW)
        reps_df.to_excel(w, index=False, sheet_name=SHEET_REPS)
        pd.DataFrame([{
            "total_reps": m["n_reps"],
            "cadence_spm": m["cadence_spm"],
            "mean_amp_g": m["mean_amp_g"],
            "sd_amp_g": m["sd_amp_g"],
            "ipi_mean_s": m["ipi_mean_s"],
            "ipi_sd_s": m["ipi_sd_s"],
            "periodicity": m["periodicity"]
        }]).to_excel(w, index=False, sheet_name=SHEET_SUM)
    return len(reps_df), m, reps_df


# =========================
# MAIN
# =========================
def main():
    timestamp = now_ts()
    out_path = Path(__file__).parent / f"imu_seated_march_SERIAL_{timestamp}.xlsx"
    print(f"Saving raw+metrics to: {out_path}")

    rows = []
    last_save = time.time()
    last_reported_reps = 0 

    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    
    try:
        s = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Connected. Waiting for V1_ACCEL data...")

        while True:
            if s.in_waiting > 0:
                line = s.readline().decode("utf-8", errors="ignore").strip()
                
                parts = []
                if line.startswith("V1_ACCEL:"):
                    clean_line = line.replace("V1_ACCEL:", "")
                    temp_parts = clean_line.split(',')
                    if len(temp_parts) == LEN_V1:
                        parts = temp_parts

                if len(parts) == LEN_V1:
                    rows.append(parts)

                    # Save & Process
                    if (time.time() - last_save >= SAVE_EVERY_SECONDS) or (len(rows) >= SAVE_EVERY_ROWS):
                        df_new = pd.DataFrame(rows, columns=HEADERS)
                        for c in HEADERS: df_new[c] = pd.to_numeric(df_new[c], errors="coerce")

                        if out_path.exists():
                            try:
                                df_exist = pd.read_excel(out_path, sheet_name=SHEET_RAW, engine="openpyxl")
                            except:
                                df_exist = pd.DataFrame(columns=HEADERS)
                            raw_df = pd.concat([df_exist, df_new], ignore_index=True)
                        else:
                            raw_df = df_new

                        reps_count, m, reps_df = save_excel(out_path, raw_df)
                        rows.clear()
                        last_save = time.time()

                        # Print flags for any NEW steps
                        if not reps_df.empty and reps_count > last_reported_reps:
                            new = reps_df.iloc[last_reported_reps:reps_count]
                            for _, r in new.iterrows():
                                t_s = float(r["t_ms"])/1000.0
                                # Print status
                                status = "OK" if r['good_rep'] else "WEAK"
                                print(
                                    f"[STEP] t={t_s:.2f}s | "
                                    f"Amp={float(r['amp_g']):.4f}g | "
                                    f"Status={status}"
                                )
                            last_reported_reps = reps_count

                        # Console Summary
                        print(
                            f"\rSaved: rows={len(raw_df)} | reps={reps_count} | "
                            f"Avg Amp={m['mean_amp_g']:.4f} g    ",
                            end=''
                        )

    except KeyboardInterrupt:
        print("\nStopping...")
        s.close()
    finally:
        if 's' in locals() and s.is_open: s.close()
        
        # Final Save
        if rows:
            df_new = pd.DataFrame(rows, columns=HEADERS)
            for c in HEADERS: df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
            if out_path.exists():
                try:
                    old = pd.read_excel(out_path, sheet_name=SHEET_RAW)
                    raw = pd.concat([old, df_new])
                except: raw = df_new
            else: raw = df_new
            save_excel(out_path, raw)
            print("\nFinal Excel file saved to:", out_path)

if __name__ == "__main__":
    main()