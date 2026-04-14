import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


WINDOW_SIZE = 10
DATA_DIR = Path("data/processed")
OUT_DIR = Path("results/tcn_sampler")

TRAIN_PATH = DATA_DIR / f"gru_train_w{WINDOW_SIZE}.npz"
VAL_PATH = DATA_DIR / f"gru_val_w{WINDOW_SIZE}.npz"
TEST_PATH = DATA_DIR / f"gru_test_w{WINDOW_SIZE}.npz"
META_PATH = DATA_DIR / f"gru_metadata_w{WINDOW_SIZE}.json"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq, feat) -> (batch, feat, seq)
        x = self.net(x)
        x = self.head(x).squeeze(1)
        return x


def load_split(path: Path):
    data = np.load(path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    return X, y


def load_all_data():
    X_train, y_train = load_split(TRAIN_PATH)
    X_val, y_val = load_split(VAL_PATH)
    X_test, y_test = load_split(TEST_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


def make_train_loader_with_sampler(X, y, batch_size=2048):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    y_int = y.astype(np.int64)
    class_counts = np.bincount(y_int, minlength=2)

    neg_count = max(class_counts[0], 1)
    pos_count = max(class_counts[1], 1)

    class_weights = np.array(
        [1.0 / neg_count, 1.0 / pos_count],
        dtype=np.float64,
    )
    sample_weights = class_weights[y_int]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def make_eval_loader(X, y, batch_size=2048):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_probs(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            probs = torch.sigmoid(logits)

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    avg_loss = total_loss / total_count
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels).astype(np.int64)

    return avg_loss, y_true, y_prob


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


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    best = None

    for threshold in thresholds:
        result = metrics_at_threshold(y_true, y_prob, threshold)
        if best is None or result["f1"] > best["f1"]:
            best = result

    return best


def save_loss_curve(history, out_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TCN Sampler Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_ap_curve(history, out_path):
    epochs = list(range(1, len(history["val_ap"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_ap"], label="val_ap")
    plt.xlabel("Epoch")
    plt.ylabel("Average Precision (PR-AUC)")
    plt.title("TCN Sampler Validation PR-AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_pr_curve(y_true, y_prob, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.6f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("TCN Sampler Test Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_all_data()

    train_loader = make_train_loader_with_sampler(X_train, y_train, batch_size=2048)
    val_loader = make_eval_loader(X_val, y_val, batch_size=2048)
    test_loader = make_eval_loader(X_test, y_test, batch_size=2048)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNClassifier(
        input_dim=X_train.shape[2],
        channels=64,
        dropout=0.2,
    ).to(device)

    raw_pos_count = float(y_train.sum())
    raw_neg_count = float(len(y_train) - raw_pos_count)
    raw_pos_weight = raw_neg_count / max(raw_pos_count, 1.0)

    clipped_pos_weight_value = min(raw_pos_weight, 100.0)
    pos_weight = torch.tensor(clipped_pos_weight_value, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 12
    best_val_ap = -1.0
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ap": [],
        "val_best_f1": [],
        "val_best_threshold": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_count += batch_size

        train_loss = total_train_loss / total_train_count

        val_loss, val_true, val_prob = evaluate_probs(model, val_loader, criterion, device)
        val_ap = average_precision_score(val_true, val_prob)
        val_best = find_best_threshold(val_true, val_prob)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_ap"].append(float(val_ap))
        history["val_best_f1"].append(float(val_best["f1"]))
        history["val_best_threshold"].append(float(val_best["threshold"]))

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_ap={val_ap:.6f} | "
            f"val_best_f1={val_best['f1']:.6f} | "
            f"val_best_threshold={val_best['threshold']:.2f}"
        )

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = {
                "epoch": epoch,
                "val_ap": float(val_ap),
                "model_state_dict": copy.deepcopy(model.state_dict()),
            }

    torch.save(best_state, OUT_DIR / "best_model.pt")

    model.load_state_dict(best_state["model_state_dict"])

    val_loss, val_true, val_prob = evaluate_probs(model, val_loader, criterion, device)
    test_loss, test_true, test_prob = evaluate_probs(model, test_loader, criterion, device)

    best_threshold_info = find_best_threshold(val_true, val_prob)
    chosen_threshold = best_threshold_info["threshold"]

    final_val_metrics = metrics_at_threshold(val_true, val_prob, chosen_threshold)
    final_test_metrics = metrics_at_threshold(test_true, test_prob, chosen_threshold)

    final_val_ap = average_precision_score(val_true, val_prob)
    final_test_ap = average_precision_score(test_true, test_prob)

    metrics = {
        "best_epoch": int(best_state["epoch"]),
        "device": str(device),
        "window_size": WINDOW_SIZE,
        "input_dim": int(X_train.shape[2]),
        "channels": 64,
        "raw_pos_weight_train": float(raw_pos_weight),
        "clipped_pos_weight_train": float(clipped_pos_weight_value),
        "sampler_used": True,
        "chosen_threshold_from_val": float(chosen_threshold),
        "val": {
            "loss": float(val_loss),
            "ap": float(final_val_ap),
            **final_val_metrics,
        },
        "test": {
            "loss": float(test_loss),
            "ap": float(final_test_ap),
            **final_test_metrics,
        },
        "history": history,
        "feature_cols": metadata["feature_cols"],
        "train_days": metadata["train_days"],
        "val_days": metadata["val_days"],
        "test_days": metadata["test_days"],
    }

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(OUT_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    np.savez_compressed(
        OUT_DIR / "test_outputs.npz",
        y_true=test_true,
        y_prob=test_prob,
    )

    save_loss_curve(history, OUT_DIR / "loss_curve.png")
    save_ap_curve(history, OUT_DIR / "val_ap_curve.png")
    save_pr_curve(test_true, test_prob, OUT_DIR / "test_pr_curve.png")

    print("\n=== FINAL TCN SAMPLER RESULTS ===")
    print(f"Best epoch: {best_state['epoch']}")
    print(f"Chosen threshold from val: {chosen_threshold:.2f}")
    print(f"Val AP:  {final_val_ap:.6f}")
    print(f"Test AP: {final_test_ap:.6f}")
    print(f"Test precision: {final_test_metrics['precision']:.6f}")
    print(f"Test recall:    {final_test_metrics['recall']:.6f}")
    print(f"Test F1:        {final_test_metrics['f1']:.6f}")

    print("\nSaved artifacts:")
    print(OUT_DIR / "best_model.pt")
    print(OUT_DIR / "metrics.json")
    print(OUT_DIR / "history.json")
    print(OUT_DIR / "test_outputs.npz")
    print(OUT_DIR / "loss_curve.png")
    print(OUT_DIR / "val_ap_curve.png")
    print(OUT_DIR / "test_pr_curve.png")


if __name__ == "__main__":
    main()