import json
from pathlib import Path

import h5py
import napari
import numpy as np
import pandas as pd

CELL_NAMES = {
    "lymphocytes": "CD3CD20_Lymphocyte",
    "epithelium": "panCK_Epithelium",
    "plasma_cells": "MIST1_PlasmaCell",
    "leukocytes": "CD45RB_Leukocyte",
    "smooth_muscle": "aSMA_SmoothMuscle",
    "endothelium": "ERG_Endothelium",
}

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/segpath")


def visualize_filtered_objects(n_images):
    for cell_type in CELL_NAMES.keys():
        csv_path = ROOT / CELL_NAMES[cell_type] / f"{CELL_NAMES[cell_type]}_summary.csv"
        h5_dir = ROOT / CELL_NAMES[cell_type] / "data"
        df = pd.read_csv(csv_path, index_col="filename")
        df = df[df["n_filtered_indices"] > 0]
        df["filtered_indices"] = df["filtered_indices"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        i = 1
        for filename in df.index.tolist():
            if i == n_images:
                break
            h5_path = h5_dir / filename
            filtered_indices = df.loc[filename, "filtered_indices"]
            filtered_indices = [idx for idx in filtered_indices]
            with h5py.File(h5_path, "r") as f:
                img = f["images/best_crop"][:]
                pred = f["labels/postprocessed_pred"][:]
            filtered_instances = [np.unique(pred)[1:].tolist()[idx] for idx in filtered_indices]
            mask = np.isin(pred, filtered_instances)
            filtered_pred = np.where(mask, pred, 0)
            viewer = napari.Viewer()

            viewer.add_image(img, name=Path(filename).stem)
            viewer.add_labels(filtered_pred, name="filtered_indices_pred")
            viewer.add_labels(pred, name="original_pred")

            i += 1
            napari.run()


visualize_filtered_objects(n_images=10)
