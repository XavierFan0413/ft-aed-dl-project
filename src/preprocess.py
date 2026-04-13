from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = Path("data/raw/nashville_freeway_anomaly.csv")
OUT_DIR = Path("data/processed")

FEATURE_COLS = [
    "milemarker",
    "lane1_speed", "lane1_volume", "lane1_occ",
    "lane2_speed", "lane2_volume", "lane2_occ",
    "lane3_speed", "lane3_volume", "lane3_occ",
    "lane4_speed", "lane4_volume", "lane4_occ",
]

LABEL_COL = "human_label"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    df = df.sort_values(["day", "unix_time", "milemarker"]).reset_index(drop=True)

    unique_days = sorted(df["day"].unique().tolist())
    n_days = len(unique_days)

    train_days = unique_days[: int(n_days * 0.6)]
    val_days = unique_days[int(n_days * 0.6): int(n_days * 0.8)]
    test_days = unique_days[int(n_days * 0.8):]

    train_df = df[df["day"].isin(train_days)].copy()
    val_df = df[df["day"].isin(val_days)].copy()
    test_df = df[df["day"].isin(test_days)].copy()

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df[LABEL_COL].to_numpy(dtype=np.float32)

    X_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val = val_df[LABEL_COL].to_numpy(dtype=np.float32)

    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test_df[LABEL_COL].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    np.savez_compressed(
        OUT_DIR / "baseline_data.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    metadata = {
        "feature_cols": FEATURE_COLS,
        "label_col": LABEL_COL,
        "train_days": train_days,
        "val_days": val_days,
        "test_days": test_days,
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "train_pos": int(y_train.sum()),
        "val_pos": int(y_val.sum()),
        "test_pos": int(y_test.sum()),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pos_weight_train": float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)),
    }

    with open(OUT_DIR / "baseline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== PREPROCESS DONE ===")
    print("Unique days:", unique_days)
    print("Train days:", train_days)
    print("Val days:  ", val_days)
    print("Test days: ", test_days)

    print("\n=== SPLIT SIZES ===")
    print("Train:", len(train_df))
    print("Val:  ", len(val_df))
    print("Test: ", len(test_df))

    print("\n=== POSITIVE COUNTS ===")
    print("Train:", int(y_train.sum()))
    print("Val:  ", int(y_val.sum()))
    print("Test: ", int(y_test.sum()))

    print("\nSaved:")
    print(OUT_DIR / "baseline_data.npz")
    print(OUT_DIR / "baseline_metadata.json")


if __name__ == "__main__":
    main()