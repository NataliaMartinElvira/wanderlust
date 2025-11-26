# this code computes per-participant metrics

# import modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

# ==========================================
# 1. READ EXCEL FILE
# ==========================================

EXCEL_FILE = "all_participants.xlsx"   # <--- change if needed
SHEET_NAMES = ["person1", "person2", "person3"]

# canonical column names we expect (human-readable)
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

FEEDBACK_LABELS = ["none", "hips", "trunk", "rotate"]


# ==========================================
# 2. HELPER FUNCTIONS TO READ EXCEL AND PROCESS DATA
# ==========================================

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


def get_true_feedback_type(row):
    """
    Ground-truth feedback from movement:
        - good rep         -> 'none'
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
        raise ValueError(f"Row has no valid movement category: {row}")


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
    Feedback appropriateness rule:

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

    # bad rep
    if detected:
        return 1 if pred_type == true_type else 0
    else:
        return 1 if pred_type in ["none", true_type] else 0


# ==========================================
# 3. LOAD & PROCESS DATA
# ==========================================

def load_and_process_data():
    if not os.path.exists(EXCEL_FILE):
        raise FileNotFoundError(
            f"Cannot find Excel file: {EXCEL_FILE}."
        )

    combined_df = pd.DataFrame()
    print("--- LOADING DATA ---")

    for sheet in SHEET_NAMES:
        try:
            df = pd.read_excel(EXCEL_FILE, sheet_name=sheet)
            print(f"\nSheet '{sheet}' original columns:", list(df.columns))

            # smart column renaming
            raw_cols = list(df.columns)
            norm_map = {normalize_key(c): c for c in raw_cols}
            rename_dict = {}
            for norm_key, canon_name in CANONICAL_COLUMNS.items():
                if norm_key in norm_map:
                    original_name = norm_map[norm_key]
                    rename_dict[original_name] = canon_name

            df = df.rename(columns=rename_dict)
            print(f"Sheet '{sheet}' after renaming:", list(df.columns))

            missing = [c for c in CANONICAL_COLUMNS.values() if c not in df.columns]
            if missing:
                print(f"[ERROR] Sheet '{sheet}' is missing columns: {missing}")
                continue

            # binary columns
            for col in BINARY_COLS:
                df[col] = df[col].apply(to_binary)

            df["participant"] = sheet

            # feedback labels
            df["feedback_true"] = df.apply(get_true_feedback_type, axis=1)
            df["feedback_pred"] = df["feedback given by program"].apply(
                get_pred_feedback_type
            )

            combined_df = pd.concat([combined_df, df], ignore_index=True)
            print(f"[OK] Loaded sheet '{sheet}' with {len(df)} rows.")

        except Exception as e:
            print(f"[ERROR] Failed to read sheet '{sheet}': {e}")

    if combined_df.empty:
        raise RuntimeError("No data loaded from any sheet.")

    # rep detection labels
    combined_df["rep_true"] = 1  # every row is a rep that exists
    combined_df["rep_pred"] = combined_df["Counted by Program"]

    # overall feedback appropriateness
    combined_df["feedback_correct"] = combined_df.apply(feedback_correct_row, axis=1)

    return combined_df


# ==========================================
# 4. PER-PARTICIPANT METRICS
# ==========================================

def compute_per_participant_metrics(df):
    # ----- rep detection sensitivity per participant -----
    rep_metrics = []
    for pid, sub in df.groupby("participant"):
        tp = int((sub["rep_pred"] == 1).sum())
        fn = int((sub["rep_pred"] == 0).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rep_metrics.append({"participant": pid, "rep_sensitivity": sens})
    rep_df = pd.DataFrame(rep_metrics)

    # ----- feedback metrics per participant (detected reps only) -----
    fb_rows = []
    df_fb = df[df["rep_pred"] == 1].copy()  # only detected reps

    for pid, sub in df_fb.groupby("participant"):
        for cls in FEEDBACK_LABELS:
            true_cls = (sub["feedback_true"] == cls)
            pred_cls = (sub["feedback_pred"] == cls)
            tp = int((true_cls & pred_cls).sum())
            fn = int((true_cls & ~pred_cls).sum())
            fp = int((~true_cls & pred_cls).sum())
            tn = int((~true_cls & ~pred_cls).sum())

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            fb_rows.append({
                "participant": pid,
                "feedback_type": cls,
                "sensitivity": sens,
                "specificity": spec
            })

    fb_df = pd.DataFrame(fb_rows)

    # ----- overall feedback appropriateness per participant -----
    overall_fb = (
        df.groupby("participant")["feedback_correct"]
        .mean()
        .reset_index()
        .rename(columns={"feedback_correct": "feedback_appropriate"})
    )

    return rep_df, fb_df, overall_fb


# ==========================================
# 5. COMBINED FIGURE (ALL GRAPHS)
# ==========================================

def make_combined_plot(rep_df, fb_df, overall_fb):
    """
    2x2 combined figure 
    """

    green_palette = ["#2e7d32", "#66bb6a", "#a5d6a7"]  # dark, mid, light

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # ---------------------------------------------
    # Helper to add value labels to bars
    # ---------------------------------------------
    def add_bar_labels(ax, decimals=2):
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.{decimals}f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=10)

    # ---------------------------------------------
    # 1) Rep Detection Sensitivity per Participant
    # ---------------------------------------------
    ax = axes[0, 0]
    sns.barplot(
        data=rep_df,
        x="participant",
        y="rep_sensitivity",
        hue="participant",
        palette=green_palette,
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Rep Detection Sensitivity by Participant", fontsize=14)
    ax.set_ylabel("Sensitivity")
    add_bar_labels(ax)
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=15)

    # ---------------------------------------------
    # 2) Overall Feedback Appropriateness per Participant
    # ---------------------------------------------
    ax = axes[0, 1]
    sns.barplot(
        data=overall_fb,
        x="participant",
        y="feedback_appropriate",
        hue="participant",
        palette=green_palette,
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall Feedback Appropriateness by Participant", fontsize=14)
    ax.set_ylabel("Proportion Appropriate")
    add_bar_labels(ax)
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=15)

    # ---------------------------------------------
    # 3) Feedback Sensitivity per Type & Participant (Detected Reps)
    # ---------------------------------------------
    ax = axes[1, 0]
    sns.barplot(
        data=fb_df,
        x="feedback_type",
        y="sensitivity",
        hue="participant",
        palette=green_palette,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Feedback Sensitivity by Type & Participant (Detected Reps)", fontsize=14)
    ax.set_ylabel("Sensitivity")
    add_bar_labels(ax)
    ax.tick_params(axis='x', rotation=0)

    # ---------------------------------------------
    # 4) Feedback Specificity per Type & Participant (Detected Reps)
    # ---------------------------------------------
    ax = axes[1, 1]
    sns.barplot(
        data=fb_df,
        x="feedback_type",
        y="specificity",
        hue="participant",
        palette=green_palette,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Feedback Specificity by Type & Participant (Detected Reps)", fontsize=14)
    ax.set_ylabel("Specificity")
    add_bar_labels(ax)
    ax.tick_params(axis='x', rotation=0)

    # ---------------------------------------------
    # Global layout fixes
    # ---------------------------------------------
    # Make spacing better to prevent label overlap
    plt.tight_layout(pad=3.0)

    # Center the legends for the bottom plots
    axes[1, 0].legend(title="Participant", bbox_to_anchor=(0.5, -0.25), loc="upper center", ncol=3)
    axes[1, 1].legend(title="Participant", bbox_to_anchor=(0.5, -0.25), loc="upper center", ncol=3)

    # Save & show
    plt.savefig("combined_robustness_plots.png", dpi=300, bbox_inches="tight")
    print("Saved: combined_robustness_plots.png")
    plt.show()


# ==========================================
# 6. LOGISTIC REGRESSIONS (OPTIONAL STATS)
# ==========================================

def run_logistic_models(df):
    """
    Logistic regressions with participant as fixed effect.
    """

    # Rep detection ~ participant
    model_det = smf.glm(
        formula="rep_pred ~ C(participant)",
        data=df,
        family=sm.families.Binomial()
    ).fit()
    print("\n=== Logistic regression: Rep detection ~ participant ===")
    print(model_det.summary())

    # Feedback correctness ~ participant
    model_fb = smf.glm(
        formula="feedback_correct ~ C(participant)",
        data=df,
        family=sm.families.Binomial()
    ).fit()
    print("\n=== Logistic regression: Feedback correct ~ participant ===")
    print(model_fb.summary())

    return model_det, model_fb


# ==========================================
# 7. MAIN
# ==========================================

def main():
    df = load_and_process_data()

    rep_df, fb_df, overall_fb = compute_per_participant_metrics(df)

    print("\nPer-participant rep detection sensitivity:")
    print(rep_df)

    print("\nPer-participant feedback sensitivity & specificity (detected reps):")
    print(fb_df)

    print("\nPer-participant overall feedback appropriateness:")
    print(overall_fb)

    make_combined_plot(rep_df, fb_df, overall_fb)

    # Optional stats:
    run_logistic_models(df)


if __name__ == "__main__":
    main()
