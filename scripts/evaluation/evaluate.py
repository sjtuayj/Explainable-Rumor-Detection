"""
Evaluate classification accuracy on val.csv and generate a few explanation examples.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rumor_detection.detect import RumourDetector

DATA_DIR = os.environ.get("RUMOR_DATA_DIR", str(PROJECT_ROOT / "rumer2026"))


def evaluate():
    detector = RumourDetector()
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))

    print("=== Classification evaluation ===")
    preds = detector.classify_batch(val_df["text"].tolist())
    acc = accuracy_score(val_df["label"], preds)
    print(f"Val Accuracy: {acc:.4f}")
    print(classification_report(val_df["label"], preds))

    n_explain = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"\n=== Explanation examples: first {n_explain} rows ===")
    for i in range(min(n_explain, len(val_df))):
        row = val_df.iloc[i]
        label = preds[i]
        explanation = detector.explain(row["text"], label)
        correct = "OK" if label == row["label"] else "ERR"
        print(f"\n[{i + 1}] {correct} pred={label} true={row['label']}")
        print(f"  Text: {row['text'][:100]}...")
        print(f"  Explanation: {explanation}")
        time.sleep(0.5)


if __name__ == "__main__":
    evaluate()
