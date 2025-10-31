# 1_build_dataset.py
# Build a ML-ready dataset from IMU Excel logs.
# Conventions:
#   - Put your logs in ./data/raw/
#   - File names must contain the label, e.g.: shoulder_flexion_001.xlsx
#   - Output: ./data/processed/dataset_features.csv
# build_dataset.py
# Usage:
#   python build_dataset.py --data-dir ./sessions --out dataset.csv
#
# Each Excel file must have columns:
#   time_ms, acc_x_g, acc_y_g, acc_z_g, pitch_deg, roll_deg, gyr_x_dps, gyr_y_dps, gyr_z_dps
# Label is inferred from the file name prefix before the first underscore.

import argparse
from pathlib import Path
import pandas as pd
import re

EXPECTED_COLS = [
    "time_ms","acc_x_g","acc_y_g","acc_z_g",
    "pitch_deg","roll_deg","gyr_x_dps","gyr_y_dps","gyr_z_dps"
]

def infer_label(filename: str) -> str:
    # label is everything before first underscore OR entire stem if no underscore
    stem = Path(filename).stem
    parts = stem.split("_")
    return parts[0].lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="dataset.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = list(data_dir.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"No .xlsx files found in {data_dir}")

    frames = []
    for f in files:
        label = infer_label(f.name)
        df = pd.read_excel(f, engine="openpyxl")
        missing = [c for c in EXPECTED_COLS if c not in df.columns]
        if missing:
            print(f"Skipping {f} (missing columns: {missing})")
            continue
        df = df[EXPECTED_COLS].copy()
        df["label"] = label
        df["source_file"] = f.name
        frames.append(df)

    if not frames:
        raise SystemExit("No valid files assembled.")
    out = pd.concat(frames, ignore_index=True)
    # Ensure numeric types
    for c in EXPECTED_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out.to_csv(args.out, index=False)
    print(f"Saved dataset: {args.out}   rows={len(out)}   labels={out['label'].nunique()}")

if __name__ == "__main__":
    main()
