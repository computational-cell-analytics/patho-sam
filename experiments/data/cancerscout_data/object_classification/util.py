from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd

CELL_NAME_ID = {
    "tumor_cells": 1,
    "stromal_cells": 2,
    "lymphocytes": 3,
    "others": 4,
    "neutrophils": 5,
    "epithelial_cells": 6,
}

SUBTYPE_NAME_ID = {
    "acinous": 1,
    "papillary": 2,
    "solid": 3,
    "mucinous": 4,
    "lepidic": 5,
    "lepidic_mixed": 6,
    "micropapillary": 7,
}

SUBTYPE_ID_NAME = {v: k for k, v in SUBTYPE_NAME_ID.items()}
CELL_ID_NAME = {v: k for k, v in CELL_NAME_ID.items()}


def get_rf_data_cancerscout(
    csv_path: Path, data_path: Path, split: str, healthy: bool, cell_types: List, return_subtypes: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, index_col="filename")
    # breakpoint()
    df = df[df["train_eval_split"] == split]
    if not healthy:
        df = df[df["inst_dataset"] == "new_tumor"]

    num_training_objects = df[cell_types].sum(axis=1).sum()
    cell_types = [CELL_NAME_ID[key] for key in cell_types]

    chosen_samples = df.index.tolist()

    all_h5_paths = list(data_path.glob(f"{split}_models/*data/fixed_h5_files/*.h5"))

    chosen_h5_paths = [p for p in all_h5_paths if p.stem in chosen_samples]
    offset = 0
    with h5py.File(chosen_h5_paths[0], "r") as f:
        feature_dim = f["train_features"].shape[1]
        feature_dtype = f["train_features"].dtype

    features_out = np.empty((num_training_objects, feature_dim), dtype=feature_dtype)
    labels_out = np.empty((num_training_objects), dtype=np.uint8)
    subtypes_out = np.empty((num_training_objects), dtype=np.uint8)

    for volume_path in chosen_h5_paths:
        with h5py.File(volume_path, "r") as f:
            subtype = SUBTYPE_NAME_ID[df.loc[volume_path.stem, "subtype"]]
            features = f["train_features"]
            labels = f["train_labels"]
            n = features.shape[0]
            features_out[offset : offset + n] = features
            labels_out[offset : offset + n] = labels
            subtypes_out[offset : offset + n] = subtype
            offset += n

    if return_subtypes:
        return features_out, labels_out, subtypes_out
    else:
        return features_out, labels_out
