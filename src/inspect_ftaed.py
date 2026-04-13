from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/nashville_freeway_anomaly.csv")


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== COLUMNS ===")
    print(df.columns.tolist())

    print("\n=== HEAD ===")
    print(df.head())

    print("\n=== DTYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

    if "human_label" in df.columns:
        counts = df["human_label"].value_counts(dropna=False).sort_index()
        print("\n=== HUMAN_LABEL COUNTS ===")
        print(counts)
        print(f"Positive rate: {df['human_label'].mean():.8f}")

    if "crash_record" in df.columns:
        counts = df["crash_record"].value_counts(dropna=False).sort_index()
        print("\n=== CRASH_RECORD COUNTS ===")
        print(counts)
        print(f"Positive rate: {df['crash_record'].mean():.8f}")

    if "unix_time" in df.columns:
        ts = pd.to_datetime(df["unix_time"], unit="s", errors="coerce")
        print("\n=== TIME RANGE ===")
        print(ts.min(), "->", ts.max())

    if "day" in df.columns:
        print("\n=== DAY COUNTS ===")
        print(df["day"].value_counts().sort_index())

    if "milemarker" in df.columns:
        print("\n=== MILEMARKER SUMMARY ===")
        print(df["milemarker"].describe())

    feature_cols = [
        "lane1_speed", "lane1_volume", "lane1_occ",
        "lane2_speed", "lane2_volume", "lane2_occ",
        "lane3_speed", "lane3_volume", "lane3_occ",
        "lane4_speed", "lane4_volume", "lane4_occ",
    ]

    for col in feature_cols:
        if col in df.columns:
            print(f"\n=== {col.upper()} SUMMARY ===")
            print(df[col].describe())


if __name__ == "__main__":
    main()