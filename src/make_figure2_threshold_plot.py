from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("results/comparison/threshold_sweep_comparison.csv")
OUT_DIR = Path("results/final_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "figure2_threshold_vs_f1.png"

MODEL_ORDER = ["baseline", "gru", "gru_sampler", "tcn_sampler"]
MODEL_LABELS = {
    "baseline": "Baseline MLP",
    "gru": "Plain GRU",
    "gru_sampler": "GRU + sampler",
    "tcn_sampler": "TCN + sampler",
}

df = pd.read_csv(CSV_PATH)

plt.figure(figsize=(9, 5.5))

for model_name in MODEL_ORDER:
    sub = df[df["model"] == model_name].sort_values("threshold")
    plt.plot(
        sub["threshold"],
        sub["f1"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label=MODEL_LABELS[model_name],
    )

plt.xlabel("Decision threshold")
plt.ylabel("Test F1")
plt.title("Test F1 versus threshold across model variants")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.close()

print(f"Saved: {OUT_PATH}")