import argparse
import json
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from skimage.measure import regionprops
from tqdm import tqdm

CELL_NAMES = {
    "smooth_muscle": "aSMA_SmoothMuscle",
    "endothelium": "ERG_Endothelium",
    "lymphocytes": "CD3CD20_Lymphocyte",
    "epithelium": "panCK_Epithelium",
    "plasma_cells": "MIST1_PlasmaCell",
    "leukocytes": "CD45RB_Leukocyte",
}
ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/segpath")


def high_precision_instance_filter(
    inst_map: np.ndarray,
    stain_mask: np.ndarray,
    erosion_radius: int = 3,
    overlap_thresh: float = 0.7,
    min_area: int = 80,
    require_centroid_in_mask: bool = True,
    return_mask: bool = False,
):
    """
    Strict high-precision instance filter.

    Parameters
    ----------
    inst_map : (H, W) int
        Instance ID map (0 = background)
    stain_mask : (H, W) bool
        Binary stain foreground mask
    erosion_radius : int
        Defines stain core via binary erosion
    overlap_thresh : float
        Minimum overlap fraction (mask vs stain core)
    min_area : int
        Minimum instance area (pixels)
    require_centroid_in_mask : bool
        Enforce centroid inside stain core
    return_mask : bool
        If True returns filtered instance map,
        else returns selected IDs

    Returns
    -------
    np.ndarray or np.ndarray[int]
    """

    stain_mask = stain_mask.astype(bool)

    # --- stain core (high precision region) ---
    structure = np.ones((2 * erosion_radius + 1, 2 * erosion_radius + 1), dtype=bool)
    stain_core = binary_erosion(stain_mask, structure=structure)

    keep_ids = []

    for prop in regionprops(inst_map):
        inst_id = prop.label
        if inst_id == 0:
            continue

        mask = inst_map == inst_id

        # --- fast rejection ---
        area = prop.area
        if area < min_area:
            continue

        # --- overlap with core ---
        inter = mask & stain_core
        inter_area = inter.sum()
        if inter_area == 0:
            continue

        if (inter_area / (area + 1e-6)) < overlap_thresh:
            continue

        # --- centroid constraint ---
        if require_centroid_in_mask:
            cy, cx = prop.centroid  # already global image coordinates

            cy = int(round(cy))
            cx = int(round(cx))

            if cy < 0 or cy >= stain_core.shape[0] or cx < 0 or cx >= stain_core.shape[1] or not stain_core[cy, cx]:
                continue

        keep_ids.append(inst_id)

    all_instances = np.unique(inst_map).tolist()[1:]
    keep_ids_indices = [all_instances.index(keep_inst) for keep_inst in keep_ids]

    if not return_mask:
        return keep_ids_indices

    # rebuild filtered map
    out = np.zeros_like(inst_map)
    for i in keep_ids:
        out[inst_map == i] = i

    return out


def process_sample(file_path, cell_type):
    with h5py.File(file_path, "r") as f:
        pred = f["labels/postprocessed_pred"][:]
        bin_label = f["labels/best_crop"][:]

    if cell_type == "CD3CD20_Lymphocyte":
        require_centroid_in_mask = False
        overlap_thresh = 0.3
    else:
        require_centroid_in_mask = True
        overlap_thresh = 0.75

    overlap_thresh = 0.9 if cell_type == "panCK_Epithelium" else overlap_thresh

    keep_id_indices = high_precision_instance_filter(
        pred, bin_label, overlap_thresh=overlap_thresh, require_centroid_in_mask=require_centroid_in_mask
    )
    return file_path.name, keep_id_indices, len(keep_id_indices)


def reset_files(cell_type):
    csv_path = Path(ROOT) / cell_type / f"{cell_type}_summary.csv"
    df = pd.read_csv(csv_path, index_col="filename")

    df["predicted"] = df["predicted"].astype(bool)
    filtered_df = df[df["predicted"]]
    file_paths = [ROOT / cell_type / "data" / filename for filename in filtered_df.index.tolist()]

    _process_sample = partial(process_sample, cell_type=cell_type)
    with Pool(cpu_count() - 2) as p:
        result = list(
            tqdm(p.imap(_process_sample, file_paths), total=len(file_paths), desc=f"Filtering for {cell_type}")
        )

    update_df = pd.DataFrame(result, columns=["filename", "filtered_indices", "n_filtered_indices"]).set_index(
        "filename"
    )
    df.loc[update_df.index, ["filtered_indices", "n_filtered_indices"]] = update_df
    df["filtered_indices"] = df["filtered_indices"].apply(lambda x: json.dumps(x) if isinstance(x, list) else pd.NA)
    print(df["n_filtered_indices"].sum(), cell_type, "\n")

    df.to_csv(csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", default=None)
    args = parser.parse_args()
    if args.cell_type is not None:
        reset_files(CELL_NAMES[args.cell_type])
    else:
        for cell_name in CELL_NAMES.values():
            reset_files(cell_name)


if __name__ == "__main__":
    main()
