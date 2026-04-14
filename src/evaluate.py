import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.train_baseline import MLPBaseline
from src.train_gru import GRUClassifier


ROOT = Path(".")
RESULTS_DIR = ROOT / "results"
PROCESSED_DIR = ROOT / "data" / "processed"


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


def save_pr_curve(y_true, y_prob, out_path, title):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.6f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_baseline():
    data_path = PROCESSED_DIR / "baseline_data.npz"
    metrics_path = RESULTS_DIR / "baseline" / "metrics.json"
    checkpoint_path = RESULTS_DIR / "baseline" / "best_model.pt"

    data = np.load(data_path)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    with open(metrics_path, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)

    threshold = float(saved_metrics["chosen_threshold_from_val"])

    model = MLPBaseline(input_dim=X_test.shape[1])

    return model, checkpoint_path, X_test, y_test, threshold, "baseline"


def load_gru():
    data_path = PROCESSED_DIR / "gru_test_w10.npz"
    metrics_path = RESULTS_DIR / "gru" / "metrics.json"
    checkpoint_path = RESULTS_DIR / "gru" / "best_model.pt"

    data = np.load(data_path)
    X_test = data["X"].astype(np.float32)
    y_test = data["y"].astype(np.int64)

    with open(metrics_path, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)

    threshold = float(saved_metrics["chosen_threshold_from_val"])
    hidden_dim = int(saved_metrics.get("hidden_dim", 64))

    model = GRUClassifier(
        input_dim=X_test.shape[2],
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.2,
    )

    return model, checkpoint_path, X_test, y_test, threshold, "gru"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["baseline", "gru"],
        required=True,
        help="Which saved model to evaluate.",
    )
    args = parser.parse_args()

    if args.model == "baseline":
        model, checkpoint_path, X_test, y_test, threshold, model_name = load_baseline()
    else:
        model, checkpoint_path, X_test, y_test, threshold, model_name = load_gru()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    X_tensor = torch.from_numpy(X_test).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        y_prob = torch.sigmoid(logits).cpu().numpy()

    ap = average_precision_score(y_test, y_prob)
    threshold_metrics = metrics_at_threshold(y_test, y_prob, threshold)

    out_dir = RESULTS_DIR / model_name

    reproduced_metrics = {
        "model": model_name,
        "device": str(device),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "ap": float(ap),
        **threshold_metrics,
    }

    with open(out_dir / "reproduced_metrics.json", "w", encoding="utf-8") as f:
        json.dump(reproduced_metrics, f, indent=2)

    save_pr_curve(
        y_test,
        y_prob,
        out_dir / "reproduced_test_pr_curve.png",
        f"{model_name.upper()} Reproduced Test Precision-Recall Curve",
    )

    np.savez_compressed(
        out_dir / "reproduced_test_outputs.npz",
        y_true=y_test,
        y_prob=y_prob,
    )

    print("\n=== REPRODUCED EVALUATION ===")
    print(f"Model: {model_name}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"AP: {ap:.6f}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Precision: {threshold_metrics['precision']:.6f}")
    print(f"Recall:    {threshold_metrics['recall']:.6f}")
    print(f"F1:        {threshold_metrics['f1']:.6f}")

    print("\nSaved:")
    print(out_dir / "reproduced_metrics.json")
    print(out_dir / "reproduced_test_pr_curve.png")
    print(out_dir / "reproduced_test_outputs.npz")


if __name__ == "__main__":
    main()