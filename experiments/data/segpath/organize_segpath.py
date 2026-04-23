from pathlib import Path
from natsort import natsorted
import h5py
from scipy.ndimage import label as cc_label
from tqdm import tqdm
import numpy as np
import os
from multiprocessing import Pool
import pandas as pd
import argparse

from torch_em.data.datasets.histopathology.segpath import get_segpath_paths

from util import CELL_NAMES, ROOT


def get_best_crop(mask, crop_size=512):
    # ensure binary (or keep weights if you want weighted foreground)
    mask = (mask > 0).astype(np.uint8)

    k = crop_size

    # integral image with padding
    INTEGRAL = np.pad(mask, ((1, 0), (1, 0)), mode='constant').cumsum(0).cumsum(1)

    # compute all window sums vectorized
    # S[y, x] = sum over window starting at (y, x)
    S = (
        INTEGRAL[k:, k:]     # bottom right
        - INTEGRAL[:-k, k:]  # top right
        - INTEGRAL[k:, :-k]  # bottom left
        + INTEGRAL[:-k, :-k]  # top left
    )

    # find best location
    y, x = np.unravel_index(np.argmax(S), S.shape)

    return y, x


def check_mask(volume_path):
    sample_coords = "_".join(volume_path.stem.split("_")[3:5])
    wsi_no = volume_path.stem.split("_")[2]
    filename = volume_path.name
    with h5py.File(volume_path, 'a') as f:
        mask = f['labels/mask'][:]
        img = f['images/raw'][:]
        if not mask.any():
            num = 0
            y, x = None, None
            crop_mask, crop_img = None, None
        else:
            y, x = get_best_crop(mask)
            crop_mask = mask[y:y+512, x:x+512]
            crop_img = img[y:y+512, x:x+512]
            label, num = cc_label(crop_mask)

        if crop_mask is not None and crop_img is not None:
            if "images/best_crop" not in f:
                f.create_dataset(name="images/best_crop", data=crop_img, compression="gzip")
            if "labels/best_crop" not in f:
                f.create_dataset(name="labels/best_crop", data=crop_mask, compression="gzip")

    return wsi_no, sample_coords, num, y, x, filename


def get_statistics(path, cell_type):
    paths = get_segpath_paths(
        path=path,
        cell_types=cell_type,
        split="train",
        download=True
    )

    cell_type = CELL_NAMES[cell_type]

    cell_type_dir = Path(paths[0]).parent.parent
    volume_paths = natsorted((cell_type_dir / "data").glob("*.h5"))

    with Pool(os.cpu_count()-2) as p:
        result = list(tqdm(p.imap(check_mask, volume_paths), total=len(volume_paths)))

    df = pd.DataFrame(result, columns=["WSI_number", "sample_coords", "num_instances", "best_y", "best_x", "filename"])

    organ_df_path = Path(path) / "segpath_organs.csv"

    if organ_df_path.exists():
        organ_df = pd.read_csv(organ_df_path, header=1)
        mapping = organ_df.drop_duplicates("TMA number").set_index("TMA number")["tumour type"]
        df["tumour_type"] = df["WSI_number"].map(mapping)
    else:
        print("Missing organ mapping file. Could not assign tumour entities to samples.")

    csv_outpath = cell_type_dir / f"{cell_type}_summary.csv"
    df.to_csv(csv_outpath, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", type=str)
    parser.add_argument("--path", type=str, default=ROOT)
    args = parser.parse_args()
    get_statistics(path=args.path, cell_type=args.cell_type)


if __name__ == "__main__":
    main()
