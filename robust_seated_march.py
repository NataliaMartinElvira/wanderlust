import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

# MAKE SURE THIS NAME MATCHES YOUR FILE EXACTLY
PARTICIPANT_FILES = [
    "imu_seated_march_20251124_201439.xlsx"
]

SHEET_NAME = "reps"

def load_and_process_data():
    combined_df = pd.DataFrame()
    print("--- LOADING DATA ---")
    
    for filename in PARTICIPANT_FILES:

        if not os.path.exists(filename):
            print(f" [CRITICAL ERROR] Cannot find file: {filename}")
            print(" -> Please check the filename and ensure it is in the same folder.")
            continue

        try:
            df = pd.read_excel(filename, sheet_name=SHEET_NAME)
            
            # LABELING: First 10 rows = Good (1), Rest = Bad (0)
            df['actual_label'] = 0
            if len(df) >= 10:
                df.iloc[:10, df.columns.get_loc('actual_label')] = 1
            else:
                df['actual_label'] = 1

            # PREDICTION: Low amplitude = Bad (0), Not-low amplitude = Good (1)
            def parse_bool(val):
                return not (str(val).upper() == "FALSE" or val is False)

            df['amp_low_bool'] = df['amp_low'].apply(parse_bool)
            df['system_pred'] = np.where(df['amp_low_bool'] == False, 1, 0)

            combined_df = pd.concat([combined_df, df], ignore_index=True)
            print(f" [OK] Loaded: {filename} ({len(df)} rows)")
            
        except Exception as e:
            print(f" [ERROR] Failed to read {filename}: {e}")

    return combined_df


def analyze_robustness():
    df = load_and_process_data()
    
    if df.empty:
        print("\n❌ ANALYSIS CANNOT BE COMPLETED: No data loaded.")
        return

    y_true = df['actual_label']
    y_pred = df['system_pred']

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # REPORT TEXT
    report_text = (
        "\n" + "="*50 + "\n"
        f" AUTOMATIC ROBUSTNESS REPORT\n"
        f"="*50 + "\n"
        f"Total Samples: {len(df)}\n"
        f" > True GOOD (First 10): {sum(y_true == 1)}\n"
        f" > True BAD  (Remaining): {sum(y_true == 0)}\n"
        f"-" * 50 + "\n"
        f"ACCURACY:      {accuracy:.2%}\n"
        f"SENSITIVITY:   {sensitivity:.2%}\n"
        f"SPECIFICITY:   {specificity:.2%}\n"
        f"-" * 50 + "\n"
        f" > True Positives:  {tp}\n"
        f" > True Negatives:  {tn}\n"
        f" > False Positives: {fp}\n"
        f" > False Negatives: {fn}\n"
        f"="*50 + "\n"
    )

    print(report_text)

    with open("robustness_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(">> Text report saved as: robustness_report.txt")

    # ==========================================
    #    PROFESSIONAL-STYLE GRAPHICS
    # ==========================================

    plt.figure(figsize=(12, 5))

    # --- CONFUSION MATRIX ---
    plt.subplot(1, 2, 1)
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='GnBu',   # Professional blue-green palette
        cbar=False,
        xticklabels=['Pred: BAD', 'Pred: GOOD'],
        yticklabels=['Actual: BAD', 'Actual: GOOD']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Prediction')

    # --- METRICS BARPLOT ---
    plt.subplot(1, 2, 2)
    metrics = ['Accuracy', 'Sensitivity', 'Specificity']
    values = [accuracy, sensitivity, specificity]

    professional_colors = ['#1F3B73', '#2A6F97', '#74A3C7']  # Elegant blue tones

    bars = plt.bar(metrics, values, color=professional_colors)
    plt.ylim(0, 1.1)
    plt.title('Performance Metrics')

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            yval + 0.02,
            f"{yval:.1%}",
            ha='center', va='bottom'
        )

    plt.tight_layout()

    plt.savefig("robustness_plot.png", dpi=300)
    print(">> Plot saved as: robustness_plot.png")

    plt.show()


if __name__ == "__main__":
    analyze_robustness()
