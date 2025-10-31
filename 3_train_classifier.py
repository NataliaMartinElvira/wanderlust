# train_classifier.py
# Usage:
#   python train_classifier.py --in features.csv --model model.pkl --featlist features.txt
#
# Prints a classification report and cross-validated accuracy.

import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", default="model.pkl")
    ap.add_argument("--featlist", default="features.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    y = df["label"].astype(str)
    X = df.drop(columns=["label","source_file","t_start_s","t_end_s"], errors="ignore")
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    print("Test set performance:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    joblib.dump({"model": clf, "features": feature_names}, args.model)
    with open(args.featlist, "w") as f:
        f.write("\n".join(feature_names))
    print(f"Saved model → {args.model}")
    print(f"Saved feature list → {args.featlist}")

if __name__ == "__main__":
    main()
