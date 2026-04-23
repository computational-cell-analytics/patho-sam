import pandas as pd
from tqdm import tqdm
from add_filenames_to_df import CELL_NAMES, ROOT


def add_organs_to_df():
    organ_df_path = ROOT / "segpath_organs.csv"
    organ_df = pd.read_csv(organ_df_path, header=1)

    mapping = organ_df.drop_duplicates("TMA number").set_index("TMA number")["tumour type"]

    for cell_type in tqdm(CELL_NAMES.values()):
        cell_type_csv = ROOT / cell_type / f"{cell_type}_summary.csv"
        cell_type_df = pd.read_csv(cell_type_csv)
        if "tumour_type" in cell_type_df.keys():
            print(f"Tumour type already exists in {cell_type_csv}")
            continue
        cell_type_df["tumour_type"] = cell_type_df["WSI_number"].map(mapping)
        cell_type_df.to_csv(cell_type_csv)
        print(f"Saved new csv at {cell_type_csv}")

add_organs_to_df()