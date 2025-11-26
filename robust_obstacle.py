import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# PUT YOUR OBSTACLE SESSION FILES HERE
PARTICIPANT_FILES = [
    "session_obstacle_FINAL20251126_131805.xlsx",
    "session_obstacle_FINAL20251126_132020.xlsx",
    "session_obstacle_FINAL20251126_132538.xlsx"
]

# The sheet name in the Obstacle Code is "Step_Analysis"
SHEET_NAME = "Step_Analysis"

def load_and_process_data():
    combined_df = pd.DataFrame()
    print("--- LOADING OBSTACLE DATA ---")
    
    for filename in PARTICIPANT_FILES:
        if not os.path.exists(filename):
            print(f" [ERROR] File not found: {filename}")
            continue

        try:
            df = pd.read_excel(filename, sheet_name=SHEET_NAME)
            
            # --- 1. GROUND TRUTH LABELING (MANUAL LOGIC) ---
            # Logic: 
            #  - First 10 reps (0-9)   = GOOD (1)
            #  - Next 5 reps (10-14)   = BAD (0) -> Amplitude Fail
            #  - Next 5 reps (15-19)   = BAD (0) -> Height Fail
            
            df['actual_label'] = 0 # Default to Bad
            
            # Mark first 10 as Good
            if len(df) >= 10:
                df.iloc[:10, df.columns.get_loc('actual_label')] = 1
            else:
                # Fallback if fewer than 10 reps recorded
                df['actual_label'] = 1 

            # --- 2. SYSTEM PREDICTION LOGIC ---
            # For Obstacle, a step is Good (1) ONLY if BOTH h_ok AND a_ok are True.
            # We need to handle Excel boolean formats (TRUE/FALSE text vs Python bool)

            def parse_bool(val):
                # Returns True if the cell contains True/VERDADERO, False otherwise
                return not (str(val).strip().upper() in ["FALSE", "FALSO", "0"])

            df['h_ok_bool'] = df['h_ok'].apply(parse_bool)
            df['a_ok_bool'] = df['a_ok'].apply(parse_bool)

            # Logic: IF (Height OK) AND (Amplitude OK) -> PREDICT 1 (Good)
            df['system_pred'] = np.where((df['h_ok_bool'] & df['a_ok_bool']), 1, 0)

            # Optional: Track WHY it failed (for detailed debugging)
            # 0=Good, 1=Height Fail, 2=Amp Fail (Just for info)
            df['fail_type_pred'] = 0
            df.loc[~df['h_ok_bool'], 'fail_type_pred'] = 1 # Height fail detected
            df.loc[~df['a_ok_bool'], 'fail_type_pred'] = 2 # Amp fail detected

            combined_df = pd.concat([combined_df, df], ignore_index=True)
            print(f" [OK] Loaded: {filename} ({len(df)} rows)")
            
        except Exception as e:
            print(f" [ERROR] Reading {filename}: {e}")

    return combined_df

def analyze_robustness():
    df = load_and_process_data()
    
    if df.empty:
        print("\n NO DATA LOADED.")
        return

    y_true = df['actual_label']
    y_pred = df['system_pred']

    # Metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Report String
    report = (
        "\n" + "="*50 + "\n"
        f" ROBUSTNESS REPORT - OBSTACLE CROSSING\n"
        f"="*50 + "\n"
        f"Total Samples: {len(df)}\n"
        f" > Actual GOOD (Target: 10/file): {sum(y_true == 1)}\n"
        f" > Actual BAD  (Target: 10/file): {sum(y_true == 0)}\n"
        f"-" * 50 + "\n"
        f"ACCURACY:      {accuracy:.2%}\n"
        f"SENSITIVITY:   {sensitivity:.2%} (Detecting Good Steps)\n"
        f"SPECIFICITY:   {specificity:.2%} (Detecting Bad Steps)\n"
        f"-" * 50 + "\n"
        f" > True Positives:  {tp}\n"
        f" > True Negatives:  {tn}\n"
        f" > False Positives: {fp} (Safety Risk: Bad step marked Good)\n"
        f" > False Negatives: {fn} (Frustration: Good step marked Bad)\n"
        f"="*50 + "\n"
    )
    print(report)
    
    with open("robustness_obstacle_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # --- PLOTTING ---
    plt.figure(figsize=(12, 5))

    # 1. Confusion Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=['Pred: BAD', 'Pred: GOOD'],
                yticklabels=['Actual: BAD', 'Actual: GOOD'])
    plt.title('Confusion Matrix (Obstacle)')
    plt.ylabel('Ground Truth')
    plt.xlabel('Algorithm Output')

    # 2. Metrics Bar
    plt.subplot(1, 2, 2)
    metrics = ['Accuracy', 'Sensitivity', 'Specificity']
    vals = [accuracy, sensitivity, specificity]
    bars = plt.bar(metrics, vals, color=['#E65100', '#FB8C00', '#FFB74D'])
    plt.ylim(0, 1.1)
    plt.title('Performance Metrics')
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y + 0.02, f"{y:.1%}", ha='center')

    plt.tight_layout()
    plt.savefig("robustness_obstacle_plot.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    analyze_robustness()