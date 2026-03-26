import pandas as pd
from pathlib import Path


def main():
    data_dir = Path("data/raw")
    csv_files = sorted(data_dir.rglob("*.csv"))

    print(f"Found {len(csv_files)} CSV file(s) under data/raw")
    for file in csv_files:
        print(f"- {file}")

    if not csv_files:
        print("\nNo CSV files found yet. Put the FT-AED dataset into data/raw first.")
        return

    df = pd.read_csv(csv_files[0])

    print("\nUsing first file:", csv_files[0])
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes)

    print("\nTop missing-value counts:")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    for col in ["human_label", "crash_record", "timestamp", "time", "lane", "milemarker", "node_id"]:
        if col in df.columns:
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts(dropna=False).head(20))


if __name__ == "__main__":
    main()
