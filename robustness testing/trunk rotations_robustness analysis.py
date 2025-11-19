#!/usr/bin/env python3
"""
trunk_robustness_analysis.py

Offline robustness analysis for the seated trunk rotation system.

Inputs:
- trunk_samples_YYYYMMDD_HHMMSS.csv   (from logging acquisition script)
- trunk_events_YYYYMMDD_HHMMSS.csv    (from logging acquisition script)
- manual_labels.csv                   (hand-labeled per-rep info)

Manual labels CSV format (create this in Excel → Save as CSV):
    rep_id,side,start_t,end_t,good_rep,hips_moved,trunk_bent,too_fast,too_slow

- rep_id      : integer or string identifier
- side        : "L" or "R" (optional but informative)
- start_t     : rep start time in seconds (should match t_dev from logs)
- end_t       : rep end time in seconds
- good_rep    : 1 if this is a rep that SHOULD be counted, else 0
- hips_moved  : 1 if hips clearly moved too much, else 0
- trunk_bent  : 1 if trunk posture clearly bad (bending), else 0
- too_fast    : 1 if rep is judged too fast, else 0
- too_slow    : 1 if rep is judged too slow, else 0

The script prints:
- Rep detection metrics (sensitivity/specificity)
- HIPS_STILL vs hips_moved confusion
- UPRIGHT vs trunk_bent confusion
- REP_TOO_FAST / REP_TOO_SLOW vs manual too_fast / too_slow

Usage (example):
    python trunk_robustness_analysis.py \
        --samples trunk_samples_20250101_101500.csv \
        --events  trunk_events_20250101_101500.csv \
        --labels  manual_labels_participant01.csv
"""

import csv
import argparse
from collections import defaultdict

def read_events(events_path):
    """
    Read events CSV and return a list of dicts:
        { 't': float, 'event': str }
    """
    events = []
    with open(events_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["t_dev"])
                ev = row["event_name"].strip().upper()
            except (KeyError, ValueError):
                continue
            events.append({"t": t, "event": ev})
    return events

def read_labels(labels_path):
    """
    Read manual labels CSV and return a list of dicts.
    """
    labels = []
    with open(labels_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rep_id = row["rep_id"]
                side   = row.get("side", "").strip().upper()
                start_t = float(row["start_t"])
                end_t   = float(row["end_t"])
                good_rep   = int(row["good_rep"])
                hips_moved = int(row["hips_moved"])
                trunk_bent = int(row["trunk_bent"])
                too_fast   = int(row["too_fast"])
                too_slow   = int(row["too_slow"])
            except (KeyError, ValueError) as e:
                print(f"Skipping label row due to parse error: {row} ({e})")
                continue

            labels.append({
                "rep_id": rep_id,
                "side": side,
                "start_t": start_t,
                "end_t": end_t,
                "good_rep": good_rep,
                "hips_moved": hips_moved,
                "trunk_bent": trunk_bent,
                "too_fast": too_fast,
                "too_slow": too_slow,
            })
    return labels

def events_in_window(events, t_start, t_end):
    """Return a list of event names that occurred between t_start and t_end (inclusive)."""
    return [e["event"] for e in events if (e["t"] >= t_start and e["t"] <= t_end)]

def compute_confusion(manual_positive, system_positive):
    """
    Given 0/1 arrays for manual and system (same length),
    return TP, FP, TN, FN and sensitivity/specificity.
    """
    assert len(manual_positive) == len(system_positive)
    TP = FP = TN = FN = 0
    for m, s in zip(manual_positive, system_positive):
        if m == 1 and s == 1:
            TP += 1
        elif m == 0 and s == 1:
            FP += 1
        elif m == 0 and s == 0:
            TN += 1
        elif m == 1 and s == 0:
            FN += 1

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else None
    specificity = TN / (TN + FP) if (TN + FP) > 0 else None
    return TP, FP, TN, FN, sensitivity, specificity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True,
                        help="Path to trunk_samples_YYYYMMDD_HHMMSS.csv (not strictly needed but nice to have).")
    parser.add_argument("--events", required=True,
                        help="Path to trunk_events_YYYYMMDD_HHMMSS.csv")
    parser.add_argument("--labels", required=True,
                        help="Path to manual_labels.csv (hand-labeled per rep).")
    args = parser.parse_args()

    # We don't actually use samples for metrics below yet, but we parse the path
    # so you remember to keep them associated.
    samples_path = args.samples
    events_path  = args.events
    labels_path  = args.labels

    print(f"Reading events from: {events_path}")
    events = read_events(events_path)
    print(f"  Loaded {len(events)} events.\n")

    print(f"Reading manual labels from: {labels_path}")
    labels = read_labels(labels_path)
    print(f"  Loaded {len(labels)} labeled reps.\n")

    if not labels:
        print("No labels found, nothing to analyze.")
        return

    # --- For each rep, derive system flags based on events in [start_t, end_t] ---
    rep_manual_good      = []
    rep_system_rep       = []

    rep_manual_hips      = []
    rep_system_hips_warn = []

    rep_manual_trunk     = []
    rep_system_upright   = []

    rep_manual_fast      = []
    rep_system_fast      = []

    rep_manual_slow      = []
    rep_system_slow      = []

    print("Per-rep overview:")
    print("rep_id | t_start-t_end | good | events")

    for lab in labels:
        rep_id    = lab["rep_id"]
        side      = lab["side"]
        t_start   = lab["start_t"]
        t_end     = lab["end_t"]
        good_rep  = lab["good_rep"]
        hips      = lab["hips_moved"]
        trunk     = lab["trunk_bent"]
        too_fast  = lab["too_fast"]
        too_slow  = lab["too_slow"]

        evs = events_in_window(events, t_start, t_end)
        evs_unique = sorted(set(evs))

        # System flags
        sys_rep   = 1 if "REP_REACHED" in evs_unique else 0
        sys_hips  = 1 if "HIPS_STILL"  in evs_unique else 0
        sys_upr   = 1 if "UPRIGHT"     in evs_unique else 0
        sys_fast  = 1 if "REP_TOO_FAST" in evs_unique else 0
        sys_slow  = 1 if "REP_TOO_SLOW" in evs_unique else 0

        # Append to lists
        rep_manual_good.append(good_rep)
        rep_system_rep.append(sys_rep)

        rep_manual_hips.append(hips)
        rep_system_hips_warn.append(sys_hips)

        rep_manual_trunk.append(trunk)
        rep_system_upright.append(sys_upr)

        rep_manual_fast.append(too_fast)
        rep_system_fast.append(sys_fast)

        rep_manual_slow.append(too_slow)
        rep_system_slow.append(sys_slow)

        print(f"{rep_id:>6} | {t_start:5.2f}-{t_end:5.2f} | good={good_rep} | evs={','.join(evs_unique) or '-'}")

    print("\n--- Rep detection (good_rep vs REP_REACHED) ---")
    TP, FP, TN, FN, sens, spec = compute_confusion(rep_manual_good, rep_system_rep)
    print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    print(f"Sensitivity (detect reps that SHOULD be counted): {sens if sens is not None else 'N/A'}")
    print(f"Specificity (NOT counting non-reps):            {spec if spec is not None else 'N/A'}")

    print("\n--- Hips moving too much (hips_moved vs HIPS_STILL) ---")
    TP, FP, TN, FN, sens, spec = compute_confusion(rep_manual_hips, rep_system_hips_warn)
    print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    print("TP: hips moved & system warned (HIPS_STILL)")
    print("FP: hips OK   & system warned (false alarm)")
    print("TN: hips OK   & system quiet")
    print("FN: hips moved & system quiet (missed)")
    print(f"Sensitivity: {sens if sens is not None else 'N/A'}")
    print(f"Specificity: {spec if spec is not None else 'N/A'}")

    print("\n--- Trunk posture (trunk_bent vs UPRIGHT) ---")
    TP, FP, TN, FN, sens, spec = compute_confusion(rep_manual_trunk, rep_system_upright)
    print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    print("TP: trunk bent & system warned (UPRIGHT)")
    print("FP: trunk OK   & system warned (false alarm)")
    print("TN: trunk OK   & system quiet")
    print("FN: trunk bent & system quiet (missed)")
    print(f"Sensitivity: {sens if sens is not None else 'N/A'}")
    print(f"Specificity: {spec if spec is not None else 'N/A'}")

    print("\n--- Rep speed: too fast (too_fast vs REP_TOO_FAST) ---")
    TP, FP, TN, FN, sens, spec = compute_confusion(rep_manual_fast, rep_system_fast)
    print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    print("TP: rep judged too fast & system said REP_TOO_FAST")
    print("FP: rep normal/slow     & system said REP_TOO_FAST")
    print(f"Sensitivity: {sens if sens is not None else 'N/A'}")
    print(f"Specificity: {spec if spec is not None else 'N/A'}")

    print("\n--- Rep speed: too slow (too_slow vs REP_TOO_SLOW) ---")
    TP, FP, TN, FN, sens, spec = compute_confusion(rep_manual_slow, rep_system_slow)
    print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    print("TP: rep judged too slow & system said REP_TOO_SLOW")
    print("FP: rep normal/fast     & system said REP_TOO_SLOW")
    print(f"Sensitivity: {sens if sens is not None else 'N/A'}")
    print(f"Specificity: {spec if spec is not None else 'N/A'}")

    print("\nAnalysis complete.\n"
          "You can now adjust thresholds (e.g. MAX_PELVIS_DRIFT_DEG, MAX_COMP_ANGLE_DEG,\n"
          "REP_MIN_DURATION_S, REP_MAX_DURATION_S) to trade off sensitivity vs specificity.\n")

if __name__ == "__main__":
    main()
