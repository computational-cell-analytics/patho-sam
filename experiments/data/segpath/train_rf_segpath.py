from pathlib import Path
import h5py
import joblib
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List
from multiprocessing import cpu_count
from micro_sam.sam_annotator.object_classifier import _train_rf

from util import ROOT, CELL_TYPE_MAPPING

def get_rf_data(path: str, cell_types: List, feature_dim: int = 257, split: str = None):
    path = Path(path)

    all_features = []
    all_labels = []

    for cell_type in cell_types:
        df = pd.read_csv((path / f"{cell_type}_summary.csv"), index_col="filename")
        filtered_df = df[(df["split"] == split)]
        num_training_objects = filtered_df["training_objects"].sum()
        volume_paths = [path / "data" / filename for filename in filtered_df.index.tolist()]

        out = np.empty((num_training_objects, feature_dim), dtype=np.float32)

        offset = 0

        for volume_path in tqdm(volume_paths):
            with h5py.File(volume_path, 'r') as f:
                features = f["object_features"][:]
            n = features.shape[0]
            out[offset:offset+n] = features
            offset += n
        all_features.append(out)
        all_labels.append(np.full(out.shape[0], CELL_TYPE_MAPPING[cell_type]))

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


def train_rf(path, cell_types):
    path = Path(path)
    features, labels = get_rf_data(path, cell_types, split="train")

    rf_dir = path / "rf_models"
    rf_dir.mkdir(parents=True, exist_ok=True)

    rf = _train_rf(
        features=features,
        labels=labels,
        n_estimators=200,
        max_depth=10,
        n_jobs=cpu_count(),
    )
    rf_outpath = rf_dir / f"rf_{'_'.join(cell_types)}.joblib"
    joblib.dump(rf, rf_outpath)


def main():
    train_rf(ROOT, ["epithelium", "smooth_muscle", "lymphocytes"])
