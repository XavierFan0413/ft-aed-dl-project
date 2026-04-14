import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "comparison"

DEFAULT_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_model_outputs(model_name):
    outputs_path = RESULTS_DIR / model_name / "reproduced_test_outputs.npz"
    metrics_path = RESULTS_DIR / model_name / "metrics.json"

    data = np.load(outputs_path)
    y_true = data["y_true"].astype(np.int64)
    y_prob = data["y_prob"].astype(np.float32)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    chosen_threshold = float(metrics["chosen_threshold_from_val"])

    return y_true, y_prob, chosen_threshold


def sweep_model(model_name):
    y_true, y_prob, chosen_threshold = load_model_outputs(model_name)

    thresholds = sorted(set(DEFAULT_THRESHOLDS + [round(chosen_threshold, 2)]))

    rows = []
    for threshold in thresholds:
        row = metrics_at_threshold(y_true, y_prob, threshold)
        row["model"] = model_name
        row["is_val_chosen_threshold"] = abs(threshold - round(chosen_threshold, 2)) < 1e-9
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[
        [
            "model",
            "threshold",
            "is_val_chosen_threshold",
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
    ]

    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_df = sweep_model("baseline")
    gru_df = sweep_model("gru")

    baseline_df.to_csv(OUT_DIR / "baseline_threshold_sweep.csv", index=False)
    gru_df.to_csv(OUT_DIR / "gru_threshold_sweep.csv", index=False)

    combined_df = pd.concat([baseline_df, gru_df], ignore_index=True)
    combined_df.to_csv(OUT_DIR / "threshold_sweep_comparison.csv", index=False)

    print("\n=== BASELINE THRESHOLD SWEEP ===")
    print(baseline_df.to_string(index=False))

    print("\n=== GRU THRESHOLD SWEEP ===")
    print(gru_df.to_string(index=False))

    print("\nSaved:")
    print(OUT_DIR / "baseline_threshold_sweep.csv")
    print(OUT_DIR / "gru_threshold_sweep.csv")
    print(OUT_DIR / "threshold_sweep_comparison.csv")


if __name__ == "__main__":
    main()