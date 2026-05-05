import argparse
from pathlib import Path

import pandas as pd
from util import CELL_NAMES, ROOT


def create_data_splits(
    path, cell_type, min_instances=2, overwrite_split: bool = False, n_samples: int = None, test_fraction: float = 0.2
) -> pd.DataFrame:
    path = Path(path)
    csv_path = path / CELL_NAMES[cell_type] / f"{CELL_NAMES[cell_type]}_summary.csv"
    df = pd.read_csv(csv_path, index_col="filename").astype({"randomly_sampled": "boolean"})

    if "split" in df.keys() and not overwrite_split:
        print(f"Split for {cell_type} already exists.")
        return

    filtered_df = df[(df["n_filtered_indices"] >= min_instances) & (df["randomly_sampled"])]

    if n_samples is not None:
        filtered_df = (filtered_df.sample(len(filtered_df), random_state=42)).iloc[:n_samples]

    unique_wsis = filtered_df["WSI_number"].dropna().drop_duplicates().sample(frac=1, random_state=42)

    df["split"] = pd.Series(pd.NA, index=df.index, dtype="string")

    n_test_samples = int(len(unique_wsis) * test_fraction)

    train_wsis = unique_wsis[n_test_samples:]

    test_wsis = unique_wsis[:n_test_samples]

    train_split = filtered_df[filtered_df["WSI_number"].isin(train_wsis)].index.tolist()
    test_split = filtered_df[filtered_df["WSI_number"].isin(test_wsis)].index.tolist()

    print("-" * 50)

    print(f"Split composition for {cell_type}: \n train: {len(train_split)} \n test: {len(test_split)}")

    df.loc[train_split, "split"] = "train"

    df.loc[test_split, "split"] = "test"

    print(df.groupby("split").agg(num_objects=("n_filtered_indices", "sum")))
    return
    df.to_csv(csv_path)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=ROOT)
    args = parser.parse_args()

    if args.cell_type is not None:
        create_data_splits(path=args.input_path, cell_type=args.cell_type)
    else:
        for cell_type in CELL_NAMES.keys():
            create_data_splits(path=ROOT, cell_type=cell_type, n_samples=10000, overwrite_split=True)


if __name__ == "__main__":
    main()
