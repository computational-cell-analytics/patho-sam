import pandas as pd
from pathlib import Path

from util import CELL_NAMES, ROOT


def get_segpath_statistics(instances_count="num_instances"):
    # summary_csv_path = ROOT / "segpath_metadata" / "all_cell_types_summary.csv"
    # df = pd.DataFrame(["n_instance_containing_images", "avg_instances", "median_instances", "n_instances_lung"])
    for cell_type in CELL_NAMES.values():
        cell_type_csv = ROOT / cell_type / f"{cell_type}_summary.csv"
        cell_type_df = pd.read_csv(cell_type_csv)
        cell_type_df = cell_type_df[cell_type_df["num_instances"] > 0]
        cell_type_df_grouped = cell_type_df.groupby(["tumour_type"]).agg(
            images=("filename", "nunique"), instances=(instances_count, "sum"), avg_inst=(instances_count, "mean"),
            median_inst=(instances_count, "median"), standard_dev=(instances_count, "std"))

        print(cell_type_df_grouped.head(6))
        cell_type_df_grouped.to_csv(ROOT / cell_type / f"{cell_type}_statistics.csv")


get_segpath_statistics()
