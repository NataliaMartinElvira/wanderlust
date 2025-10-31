# extract_features.py
# Usage:
#   python extract_features.py --in dataset.csv --out features.csv --fs 10 --win 2.0 --step 1.0
#
# Features per window (for acc/gyr each axis + magnitudes + pitch/roll):
#   mean, std, min, max, rms, sma, iqr, mad, zcr, dom_freq, spec_energy

import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from pathlib import Path

NUM_COLS = [
    "acc_x_g","acc_y_g","acc_z_g",
    "gyr_x_dps","gyr_y_dps","gyr_z_dps",
    "pitch_deg","roll_deg"
]

def butter_lowpass(cut, fs, order=4):
    b, a = butter(order, cut/(fs/2.0), btype="low")
    return b, a

def magnitude(x, y, z):
    return np.sqrt(x*x + y*y + z*z)

def window_indices(n, win, step):
    i = 0
    while i + win <= n:
        yield i, i+win
        i += step

def safe_dom_freq(x, fs):
    # dominant freq via Welch; return 0 if flat
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), max(8, int(fs))))
    if len(pxx) == 0 or np.allclose(pxx, 0):
        return 0.0
    return float(f[np.argmax(pxx)])

def spec_energy(x):
    # normalized spectral energy
    if len(x) == 0: return 0.0
    X = np.fft.rfft(x - np.mean(x))
    P = np.abs(X)**2
    denom = np.sum(P)
    return float(np.sum(P/denom)) if denom > 0 else 0.0

def zcr(x):
    return float(((x[:-1] * x[1:]) < 0).sum()) / max(1, len(x)-1)

def iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))

def mad(x):
    return float(np.median(np.abs(x - np.median(x))))

def sma(x, y=None, z=None):
    # signal magnitude area
    if y is None or z is None:
        return float(np.mean(np.abs(x)))
    return float(np.mean(np.abs(x) + np.abs(y) + np.abs(z)))

def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))

def extract_window_feats(win_df, fs):
    feats = {}
    # magnitudes
    acc_mag = magnitude(win_df["acc_x_g"], win_df["acc_y_g"], win_df["acc_z_g"])
    gyr_mag = magnitude(win_df["gyr_x_dps"], win_df["gyr_y_dps"], win_df["gyr_z_dps"])

    # helpers
    series = {c: win_df[c].values.astype(float) for c in NUM_COLS}
    series["acc_mag"] = acc_mag
    series["gyr_mag"] = gyr_mag

    for name, x in series.items():
        feats[f"{name}_mean"] = float(np.mean(x))
        feats[f"{name}_std"]  = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        feats[f"{name}_min"]  = float(np.min(x))
        feats[f"{name}_max"]  = float(np.max(x))
        feats[f"{name}_rms"]  = rms(x)
        feats[f"{name}_iqr"]  = iqr(x)
        feats[f"{name}_mad"]  = mad(x)
        feats[f"{name}_zcr"]  = zcr(x)
        feats[f"{name}_domf"] = safe_dom_freq(x, fs)
        feats[f"{name}_spen"] = spec_energy(x)

    # SMA for vectors
    feats["acc_sma"] = sma(series["acc_x_g"], series["acc_y_g"], series["acc_z_g"])
    feats["gyr_sma"] = sma(series["gyr_x_dps"], series["gyr_y_dps"], series["gyr_z_dps"])
    return feats

def preprocess_resample(df, fs):
    # Ensure numeric and sort by time
    df = df.copy()
    for c in ["time_ms"] + NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time_ms"]).sort_values("time_ms")
    # Make a real time index in seconds
    t = (df["time_ms"].values - df["time_ms"].values[0]) / 1000.0
    # Build uniform time grid
    if len(t) < 3:
        return None
    t_end = t[-1]
    grid = np.arange(0, t_end, 1.0/fs)
    # Interpolate each numeric column onto the grid
    out = {"time_s": grid}
    for c in NUM_COLS:
        y = df[c].values.astype(float)
        out[c] = np.interp(grid, t, y)
    out = pd.DataFrame(out)

    # Low-pass filter accel & gyro
    b, a = butter_lowpass(cut=4.0, fs=fs, order=4)
    for c in ["acc_x_g","acc_y_g","acc_z_g","gyr_x_dps","gyr_y_dps","gyr_z_dps","pitch_deg","roll_deg"]:
        out[c] = filtfilt(b, a, out[c])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--fs", type=float, default=10.0)   # Hz
    ap.add_argument("--win", type=float, default=2.0)   # seconds
    ap.add_argument("--step", type=float, default=1.0)  # seconds
    args = ap.parse_args()

    raw = pd.read_csv(args.inp)
    labels = raw["label"].unique()
    frames = []

    for label in labels:
        sub = raw[raw["label"] == label].copy()
        # Split by source_file to avoid mixing sessions
        for src in sub["source_file"].dropna().unique():
            seg = sub[sub["source_file"] == src]
            proc = preprocess_resample(seg, fs=args.fs)
            if proc is None or len(proc) < int(args.win*args.fs):
                continue

            N = len(proc)
            W = int(args.win * args.fs)
            S = int(args.step * args.fs)

            for i0, i1 in window_indices(N, W, S):
                win = proc.iloc[i0:i1]
                feats = extract_window_feats(win, fs=args.fs)
                feats["label"] = label
                feats["source_file"] = src
                feats["t_start_s"] = float(win["time_s"].iloc[0])
                feats["t_end_s"]   = float(win["time_s"].iloc[-1])
                frames.append(feats)

    if not frames:
        raise SystemExit("No windows produced. Check your data and parameters.")

    out = pd.DataFrame(frames)
    out.to_csv(args.out, index=False)
    print(f"Saved features: {args.out}  rows={len(out)}  classes={out['label'].nunique()}")

if __name__ == "__main__":
    main()
