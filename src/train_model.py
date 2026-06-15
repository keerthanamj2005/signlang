# top-of-file helper so this works when run as module or script
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.utils import load_dataset, ensure_dir


def train():
    print("Loading dataset...")
    X, y, labels = load_dataset("dataset")
    if X.size == 0:
        print("No data found. Run data_collection first.")
        return

    # Encode labels to integers
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    y_idx = [label_to_idx[l] for l in y]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, random_state=42, stratify=y_idx
    )

    # RandomForest (works well for small datasets)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    print("Training RandomForest...")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=labels))

    ensure_dir("models")
    joblib.dump(clf, "models/rf_model.joblib")
    with open("models/labels.json", "w") as f:
        json.dump(labels, f)

    print("Model saved to models/rf_model.joblib and models/labels.json")


if __name__ == "__main__":
    train()
