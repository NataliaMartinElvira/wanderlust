# Code to compute robustness metrics for trunk rotations rep detection and feedback appropriateness.

# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# ==========================================
# 1. READ EXCEL FILE
# ==========================================

EXCEL_FILE = "all_participants.xlsx"  
SHEET_NAMES = ["person1", "person2", "person3"]

# expected columns in excel
CANONICAL_COLUMNS = {
    "repnumber": "Rep Number",
    "countedbyprogram": "Counted by Program",
    "goodrep": "Good Rep",
    "hipsmoved": "hips moved",
    "leanedforward": "leaned forward",
    "tooslow": "too slow",
    "feedbackgivenbyprogram": "feedback given by program",
    "systemfeedbackappropriate": "system feedback appropriate",
}

# columns that should be binary 0/1
BINARY_COLS = [
    "Good Rep",
    "Counted by Program",
    "hips moved",
    "leaned forward",
    "too slow",
    "system feedback appropriate",
]

# =====================================================
# 2. HELPER FUNCTIONS TO READ EXCEL AND PROCESS DATA
# =====================================================

def normalize_key(name: str) -> str:
    """Lowercase and remove ALL whitespace for robust matching."""
    return "".join(str(name).lower().split())


def to_binary(val):
    """Convert various 'yes/no, true/false, 1/0, empty' to 0/1."""
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    if s in ["1", "true", "yes", "y"]:
        return 1
    if s in ["0", "false", "no", "n", ""]:
        return 0
    # fallback: try int
    try:
        return int(s)
    except Exception:
        raise ValueError(f"Cannot convert value '{val}' to binary 0/1")


def parse_feedback_cell(cell):
    """Convert feedback cell to a list of lowercase tokens (hips, trunk, rotate)."""
    if pd.isna(cell):
        return []
    txt = str(cell).strip()
    if txt == "":
        return []
    return [p.strip().lower() for p in txt.split(",") if p.strip() != ""]


# true feedback types based on movement
def get_true_feedback_type(row):
    """
    Ground-truth feedback from movement:
        - good rep         -> 'none' (no feedback)
        - hips moved       -> 'hips'
        - leaned forward   -> 'trunk'
        - too slow         -> 'rotate'
    """
    if row["Good Rep"] == 1:
        return "none"
    elif row["hips moved"] == 1:
        return "hips"
    elif row["leaned forward"] == 1:
        return "trunk"
    elif row["too slow"] == 1:
        return "rotate"
    else:
        raise ValueError(f"Row has no valid movement category: {row}") # shouldn't happen


# feedback type given by system
def get_pred_feedback_type(cell):
    """
    Predicted feedback type from system text:
        - empty           -> 'none'
        - contains 'hips' -> 'hips'
        - contains 'trunk'-> 'trunk'
        - contains 'rotate'->'rotate'
    """
    items = parse_feedback_cell(cell)

    if len(items) == 0:
        return "none"

    for key in ["hips", "trunk", "rotate"]:
        if key in items:
            return key

    raise ValueError(f"System feedback contains unexpected value(s): {items}")


def feedback_correct_row(row):
    """
    Is feedback appropriate, based on movement type and detection? 

    - good rep (feedback_true == 'none'):
        correct feedback = 'none' (detected or not)

    - bad rep (feedback_true in {'hips','trunk','rotate'}):
        if detected (rep_pred == 1):
            correct feedback = that specific label
        if NOT detected (rep_pred == 0):
            correct feedback = EITHER 'none' OR that label
    """
    true_type = row["feedback_true"]
    pred_type = row["feedback_pred"]
    detected = (row["rep_pred"] == 1)

    if true_type == "none":  # good rep
        return 1 if pred_type == "none" else 0

    # bad rep: true_type in {'hips', 'trunk', 'rotate'}
    if detected:
        # must say the specific error
        return 1 if pred_type == true_type else 0
    else:
        # undetected: both 'none' and the correct error label are acceptable
        return 1 if pred_type in ["none", true_type] else 0


# =====================================================
# 3. LOAD & PROCESS DATA
# =====================================================

def load_and_process_data():
    if not os.path.exists(EXCEL_FILE):
        raise FileNotFoundError(
            f"Cannot find Excel file: {EXCEL_FILE} (put it in the same folder as this script)."
        )

    combined_df = pd.DataFrame()
    print("--- LOADING DATA ---")

    for sheet in SHEET_NAMES:
        try:
            df = pd.read_excel(EXCEL_FILE, sheet_name=sheet)
            print(f"\nSheet '{sheet}' original columns:", list(df.columns))

            # ----- column renaming -----
            raw_cols = list(df.columns)
            norm_map = {normalize_key(c): c for c in raw_cols}
            rename_dict = {}

            for norm_key, canon_name in CANONICAL_COLUMNS.items():
                if norm_key in norm_map:
                    original_name = norm_map[norm_key]
                    rename_dict[original_name] = canon_name

            df = df.rename(columns=rename_dict)
            print(f"Sheet '{sheet}' after renaming:", list(df.columns))

            # check that all required columns exist
            missing = [c for c in CANONICAL_COLUMNS.values() if c not in df.columns]
            if missing:
                print(f"[ERROR] Sheet '{sheet}' is missing columns: {missing}")
                continue

            # ----- convert binary columns safely -----
            for col in BINARY_COLS:
                df[col] = df[col].apply(to_binary)

            # participant label
            df["participant"] = sheet

            # Ground truth and predicted feedback types (movement-based vs system)
            df["feedback_true"] = df.apply(get_true_feedback_type, axis=1)
            df["feedback_pred"] = df["feedback given by program"].apply(
                get_pred_feedback_type
            )

            combined_df = pd.concat([combined_df, df], ignore_index=True)
            print(f"[OK] Loaded sheet '{sheet}' with {len(df)} rows.")

        except Exception as e:
            print(f"[ERROR] Failed to read sheet '{sheet}': {e}")

    return combined_df


# =====================================================
# 4. ANALYSIS
# =====================================================

def analyze():
    df = load_and_process_data()

    if df.empty:
        print("\n ANALYSIS CANNOT BE COMPLETED: No data loaded.")
        return

    # -------------------------------------------------
    # 4A. REP DETECTION (rep present vs rep missed)
    # -------------------------------------------------
    # Every row is a rep (good or bad), so ground truth = 1 for all.
    df["rep_true"] = 1 # rep exists
    df["rep_pred"] = df["Counted by Program"] # 1 = detected, 0 = missed

    y_true_rep = df["rep_true"]
    y_pred_rep = df["rep_pred"]

    tp = int((y_pred_rep == 1).sum())
    fn = int((y_pred_rep == 0).sum())
    tn = 0
    fp = 0

    cm_rep = np.array([[tn, fp],
                       [fn, tp]])

    total_reps = tp + fn
    detection_accuracy = tp / total_reps if total_reps > 0 else 0.0
    detection_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # -------------------------------------------------
    # 4B. FEEDBACK TYPE CONFUSION MATRIX ( FOR DETECTED REPS)
    # -------------------------------------------------
    labels_fb = ["none", "hips", "trunk", "rotate"]

    # Only consider reps that were detected
    df_fb = df[df["rep_pred"] == 1].copy()

    y_true_fb = df_fb["feedback_true"]
    y_pred_fb = df_fb["feedback_pred"]

    cm_fb = confusion_matrix(y_true_fb, y_pred_fb, labels=labels_fb)
    feedback_accuracy_detected = accuracy_score(y_true_fb, y_pred_fb)

    # Per-class sensitivity & specificity for feedback (on detected reps)
    feedback_metrics = {}
    total_fb = cm_fb.sum()
    for i, cls in enumerate(labels_fb):
        tp_c = cm_fb[i, i]
        fn_c = cm_fb[i, :].sum() - tp_c
        fp_c = cm_fb[:, i].sum() - tp_c
        tn_c = total_fb - tp_c - fn_c - fp_c

        sens_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        spec_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0

        feedback_metrics[cls] = {
            "sensitivity": sens_c,
            "specificity": spec_c,
        }

    # -------------------------------------------------
    # 4C. OVERALL FEEDBACK APPROPRIATENESS
    # -------------------------------------------------
    df["feedback_correct"] = df.apply(feedback_correct_row, axis=1)
    overall_feedback_appropriate = df["feedback_correct"].mean() if len(df) > 0 else 0.0

    # =================================================
    # 5. TEXT REPORT
    # =================================================

    report_lines = []

    # --- REP DETECTION ---
    report_lines.append("=" * 70)
    report_lines.append(" REP DETECTION (rep present vs rep missed)")
    report_lines.append("=" * 70)
    report_lines.append(f"Total reps (good + bad): {total_reps}")
    report_lines.append(f"Detected reps (TP):      {tp}")
    report_lines.append(f"Missed reps   (FN):      {fn}")
    report_lines.append("-" * 70)
    report_lines.append(f"Detection accuracy (TP / all reps): {detection_accuracy:.2%}")
    report_lines.append(f"Sensitivity (same as above here):   {detection_sensitivity:.2%}")
    report_lines.append("Specificity: N/A (no true negatives / no 'no-rep' tests)")
    report_lines.append("Confusion matrix (rows=true, cols=pred):")
    report_lines.append("  [[TN, FP],")
    report_lines.append("   [FN, TP]] =")
    report_lines.append(f"  [[{tn}, {fp}],")
    report_lines.append(f"   [{fn}, {tp}]]")
    report_lines.append("")

    # --- FEEDBACK (DETECTED REPS) ---
    report_lines.append("=" * 70)
    report_lines.append(" FEEDBACK TYPE (only for detected reps)")
    report_lines.append("=" * 70)
    report_lines.append(f"Number of detected reps: {len(df_fb)}")
    report_lines.append(f"Feedback accuracy on detected reps: {feedback_accuracy_detected:.2%}")
    report_lines.append("Per-class sensitivity & specificity (detected reps):")
    for cls in labels_fb:
        m = feedback_metrics[cls]
        report_lines.append(
            f"  {cls:5s} -> Sensitivity: {m['sensitivity']:.2%}, "
            f"Specificity: {m['specificity']:.2%}"
        )

    # --- OVERALL FEEDBACK APPROPRIATENESS (YOUR RULE) ---
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append(" OVERALL FEEDBACK APPROPRIATENESS")
    report_lines.append("=" * 70)
    report_lines.append(
        "Good reps: correct if feedback is 'none'.\n"
        "Bad reps & detected: correct if feedback matches the error type.\n"
        "Bad reps & NOT detected: correct if feedback is 'none' OR the error type."
    )
    report_lines.append(
        f"Overall proportion of reps with appropriate feedback: "
        f"{overall_feedback_appropriate:.2%}"
    )

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open("robustness_report_feedback.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("\n>> Text report saved as: robustness_report_feedback.txt")

    # =================================================
    # 6. PLOTS
    # =================================================

    plt.figure(figsize=(14, 6))

    # --- REP DETECTION CONFUSION MATRIX ---
    plt.subplot(1, 2, 1)
    sns.heatmap(
        cm_rep,
        annot=True,
        fmt="d",
        cmap="GnBu",
        cbar=False,
        xticklabels=["Pred: no rep", "Pred: rep detected"],
        yticklabels=["True: no rep (not used)", "True: rep present"],
    )
    plt.title("Rep Detection Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")

    # --- FEEDBACK TYPE CONFUSION MATRIX (DETECTED REPS) ---
    plt.subplot(1, 2, 2)
    sns.heatmap(
        cm_fb,
        annot=True,
        fmt="d",
        cmap="GnBu",
        cbar=False,
        xticklabels=labels_fb,
        yticklabels=labels_fb,
    )
    plt.title("Feedback Type Confusion Matrix (Detected Reps)")
    plt.ylabel("True feedback type")
    plt.xlabel("Predicted feedback type")

    plt.tight_layout()
    plt.savefig("robustness_feedback_plots.png", dpi=300)
    print(">> Plots saved as: robustness_feedback_plots.png")
    plt.show()


if __name__ == "__main__":
    analyze()
