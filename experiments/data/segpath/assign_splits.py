import argparse
from pathlib import Path

import pandas as pd
from util import CELL_NAMES, ROOT

FRACTIONS = {"train": 0.6, "calibration": 0.15, "thresholding": 0.1, "test": 0.15}


def create_data_splits(
    path,
    cell_type,
    min_instances=2,
    overwrite_split: bool = False,
    n_samples: int = None,
    test_fractions: dict = FRACTIONS,
) -> pd.DataFrame:

    path = Path(path)
    csv_path = path / CELL_NAMES[cell_type] / f"{CELL_NAMES[cell_type]}_summary.csv"
    df = pd.read_csv(csv_path, index_col="filename").astype({"randomly_sampled": "boolean"})

    if "split" in df.keys() and not overwrite_split:
        print(f"Split for {cell_type} already exists.")
        return

    filtered_df = df[df["n_filtered_indices"] >= min_instances]

    filtered_df = filtered_df.sample(frac=1, random_state=42)
    if n_samples is not None:
        filtered_df = filtered_df.iloc[:n_samples]

    unique_wsis = filtered_df["WSI_number"].dropna().drop_duplicates().sample(frac=1, random_state=42)

    df["split"] = pd.Series(pd.NA, index=df.index, dtype="string")

    start = 0

    for split, fraction in test_fractions.items():
        end = start + int(len(unique_wsis) * fraction)
        fraction_wsis = unique_wsis[start:end]
        fraction_split = filtered_df[filtered_df["WSI_number"].isin(fraction_wsis)].index.tolist()
        df.loc[fraction_split, "split"] = split
        start = end

    print("-" * 50)

    print(df.groupby("split").agg(num_objects=("n_filtered_indices", "sum")))
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
