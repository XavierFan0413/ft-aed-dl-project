from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = Path("data/raw/nashville_freeway_anomaly.csv")
OUT_DIR = Path("data/processed")

WINDOW_SIZE = 10

FEATURE_COLS = [
    "milemarker",
    "lane1_speed", "lane1_volume", "lane1_occ",
    "lane2_speed", "lane2_volume", "lane2_occ",
    "lane3_speed", "lane3_volume", "lane3_occ",
    "lane4_speed", "lane4_volume", "lane4_occ",
]

LABEL_COL = "human_label"


def get_day_splits(df: pd.DataFrame):
    unique_days = sorted(df["day"].unique().tolist())
    n_days = len(unique_days)

    train_days = unique_days[: int(n_days * 0.6)]
    val_days = unique_days[int(n_days * 0.6): int(n_days * 0.8)]
    test_days = unique_days[int(n_days * 0.8):]

    return unique_days, train_days, val_days, test_days


def build_sequences(split_df: pd.DataFrame, window_size: int):
    x_chunks = []
    y_chunks = []
    day_chunks = []
    time_chunks = []
    mile_chunks = []

    used_groups = 0
    skipped_groups = 0

    grouped = split_df.groupby(["day", "milemarker"], sort=True)

    for (day, milemarker), group in grouped:
        group = group.sort_values("unix_time").reset_index(drop=True)

        features = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        labels = group[LABEL_COL].to_numpy(dtype=np.float32)
        unix_times = group["unix_time"].to_numpy(dtype=np.int64)

        if len(group) <= window_size:
            skipped_groups += 1
            continue

        n_seq = len(group) - window_size

        group_x = np.stack(
            [features[i - window_size:i] for i in range(window_size, len(group))],
            axis=0,
        ).astype(np.float32)

        group_y = labels[window_size:].astype(np.float32)
        group_day = np.full(n_seq, int(day), dtype=np.int16)
        group_time = unix_times[window_size:].astype(np.int64)
        group_mile = np.full(n_seq, float(milemarker), dtype=np.float32)

        x_chunks.append(group_x)
        y_chunks.append(group_y)
        day_chunks.append(group_day)
        time_chunks.append(group_time)
        mile_chunks.append(group_mile)

        used_groups += 1

    if not x_chunks:
        raise ValueError("No sequences were created. Check split logic or window size.")

    X = np.concatenate(x_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    days = np.concatenate(day_chunks, axis=0)
    unix_times = np.concatenate(time_chunks, axis=0)
    milemarkers = np.concatenate(mile_chunks, axis=0)

    info = {
        "used_groups": int(used_groups),
        "skipped_groups": int(skipped_groups),
        "num_sequences": int(len(y)),
        "num_positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "window_size": int(window_size),
        "feature_dim": int(X.shape[-1]),
    }

    return X, y, days, unix_times, milemarkers, info


def save_split(split_name, split_df, window_size):
    X, y, days, unix_times, milemarkers, info = build_sequences(split_df, window_size)

    out_path = OUT_DIR / f"gru_{split_name}_w{window_size}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        day=days,
        unix_time=unix_times,
        milemarker=milemarkers,
    )

    print(f"\n=== {split_name.upper()} SEQUENCES ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"positives: {int(y.sum())}")
    print(f"positive rate: {y.mean():.8f}")
    print(f"saved: {out_path}")

    return info


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        print(f"Dataset not found: {RAW_PATH}")
        return

    df = pd.read_csv(RAW_PATH)
    df = df.sort_values(["day", "milemarker", "unix_time"]).reset_index(drop=True)

    unique_days, train_days, val_days, test_days = get_day_splits(df)

    train_df = df[df["day"].isin(train_days)].copy()
    val_df = df[df["day"].isin(val_days)].copy()
    test_df = df[df["day"].isin(test_days)].copy()

    scaler = StandardScaler()
    train_df.loc[:, FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS]).astype(np.float32)
    val_df.loc[:, FEATURE_COLS] = scaler.transform(val_df[FEATURE_COLS]).astype(np.float32)
    test_df.loc[:, FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS]).astype(np.float32)

    train_info = save_split("train", train_df, WINDOW_SIZE)
    val_info = save_split("val", val_df, WINDOW_SIZE)
    test_info = save_split("test", test_df, WINDOW_SIZE)

    metadata = {
        "window_size": WINDOW_SIZE,
        "feature_cols": FEATURE_COLS,
        "label_col": LABEL_COL,
        "unique_days": unique_days,
        "train_days": train_days,
        "val_days": val_days,
        "test_days": test_days,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "train_info": train_info,
        "val_info": val_info,
        "test_info": test_info,
    }

    meta_path = OUT_DIR / f"gru_metadata_w{WINDOW_SIZE}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== DONE ===")
    print(f"metadata saved: {meta_path}")


if __name__ == "__main__":
    main()