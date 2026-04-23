from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/segpath")

CELL_NAMES = {"smooth_muscle": "aSMA_SmoothMuscle",
              "endothelium": "ERG_Endothelium",
              "lymphocytes": "CD3CD20_Lymphocyte",
              "epithelium": "panCK_Epithelium",
              "plasma_cells": "MIST1_PlasmaCell",
              "leukocytes": "CD45RB_Leukocyte",
              }


def add_filenames():
    for cell_type in CELL_NAMES.values():
        cell_type_dir = ROOT / cell_type
        h5_dir = cell_type_dir / 'data'
        csv_path = cell_type_dir / f"{cell_type}_summary.csv"
        df = pd.read_csv(csv_path)
        df = df.set_index(["WSI_number", "sample_coords"])
        breakpoint()
    # df["filename"] = pd.Series(pd.NA, index=df.index, dtype="string")

    # mappings = {}

    # for volume_path in tqdm(h5_dir.glob("*.h5")):
    #     parts = volume_path.stem.split("_")
    #     wsi_no = int(parts[2])
    #     sample_coords = "_".join(parts[3:5])
    #     key = (wsi_no, sample_coords)
    #     if key not in df.index:
    #         raise ValueError(wsi_no, sample_coords)
    #     mappings[key] = volume_path.name

    # df["filename"] = df.index.map(mappings)
    # df.to_csv(csv_path)