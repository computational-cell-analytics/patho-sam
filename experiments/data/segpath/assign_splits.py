from pathlib import Path
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

from util import ROOT, CELL_NAMES


def create_data_splits(path, cell_type, min_instances=20, overwrite_split=False):
    path = Path(path)
    csv_path = path / CELL_NAMES[cell_type] / f"{CELL_NAMES[cell_type]}_summary.csv"
    df = pd.read_csv(csv_path, index_col="filename")

    if "split" in df.keys() and not overwrite_split:
        print(f"Split for {cell_type} already exists.")
        return

    filtered_df = df[df["training_objects"] > min_instances]

    df["split"] = pd.Series(pd.NA, index=df.index, dtype="string")

    valid_samples = filtered_df.index.tolist()

    train_split, test_split = train_test_split(valid_samples, test_size=0.2, random_state=42)

    print(f"Split composition: \n train: {len(train_split)} \n test: {len(test_split)}")

    df.loc[train_split, "split"] = "train"

    df.loc[test_split, "split"] = "test"

    df.to_csv(csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", type=str)
    args = parser.parse_args()
    create_data_splits(path=ROOT, cell_type=args.cell_type)


if __name__ == "__main__":
    main()