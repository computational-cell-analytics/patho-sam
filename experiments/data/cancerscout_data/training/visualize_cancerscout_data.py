import argparse
from pathlib import Path

import h5py
import napari
import numpy as np
import pandas as pd
from _cancerscout_dataset import get_cancerscout_dataset

from patho_sam.training import histopathology_identity

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/")


def get_cs_ds():
    return get_cancerscout_dataset(
        ROOT, split="train", patch_shape=(512, 512), label_transform=None, raw_transform=histopathology_identity
    )


SEG_ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/segpath")

CELL_NAMES = {
    "smooth_muscle": "aSMA_SmoothMuscle",
    "endothelium": "ERG_Endothelium",
    "lymphocytes": "CD3CD20_Lymphocyte",
    "epithelium": "panCK_Epithelium",
    "plasma_cells": "MIST1_PlasmaCell",
    "leukocytes": "CD45RB_Leukocyte",
}


def visualize_segpath(cell_type, path, cs=False):
    _cell_type = CELL_NAMES[cell_type]
    data_dir = path / _cell_type / "data"
    csv_path = path / _cell_type / f"{_cell_type}_summary.csv"
    df = pd.read_csv(csv_path)
    df = df[df["training_objects"] > 0]
    df = df.set_index("filename")
    predicted_samples = df.index.tolist()

    paths = [data_dir / pred_sample for pred_sample in predicted_samples]

    cs_ds = get_cs_ds()

    for (cs_img, cs_label), path in zip(cs_ds, paths):
        with h5py.File(path, "r") as f:
            pred = f["labels/postprocessed_pred"][:]
            bin_mask = f["labels/best_crop"][:]
            img = f["images/best_crop"][:]

        viewer = napari.Viewer()
        if cs:
            viewer.add_image(cs_img.numpy().transpose(1, 2, 0).astype(np.uint8), name="cs_img")
            viewer.add_labels(np.squeeze(cs_label.numpy()).astype(np.uint32), name="cs_label")
        viewer.add_image(img, name=path.stem)
        viewer.add_labels(pred, name="segpath_prediction")
        viewer.add_labels(bin_mask, name="binary label")
        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", type=str)
    parser.add_argument("--path", "-p", type=str, default=SEG_ROOT)
    args = parser.parse_args()
    visualize_segpath(args.cell_type, args.path)


if __name__ == "__main__":
    main()
