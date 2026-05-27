from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/puma")


def get_best_crop(mask, crop_size=512):
    # ensure binary (or keep weights if you want weighted foreground)
    mask = (mask > 0).astype(np.uint8)
    k = crop_size

    # integral image with padding
    INTEGRAL = np.pad(mask, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)

    # compute all window sums vectorized
    # S[y, x] = sum over window starting at (y, x)
    S = (
        INTEGRAL[k:, k:]  # bottom right
        - INTEGRAL[:-k, k:]  # top right
        - INTEGRAL[k:, :-k]  # bottom left
        + INTEGRAL[:-k, :-k]  # top left
    )

    # find best location
    y, x = np.unravel_index(np.argmax(S), S.shape)

    return y, x


def check_mask(volume_path, classes_of_interest: List = []):

    with h5py.File(volume_path, "a") as f:
        inst_mask = f["labels/instances/nuclei"][:]
        semantic = f["labels/semantic/nuclei"][:]

    sample_dict = defaultdict(dict)

    for class_of_interest in classes_of_interest:
        binary_mask = semantic == class_of_interest
        y, x = get_best_crop(binary_mask)
        crop_sem = semantic[y : y + 512, x : x + 512]
        crop_inst = inst_mask[y : y + 512, x : x + 512]
        sample_dict[volume_path.name]["best_crop"][class_of_interest] = len(
            np.unique(crop_inst[crop_sem == class_of_interest])
        )
        sample_dict[volume_path.name]["whole_img"][class_of_interest] = len(
            np.unique(inst_mask[semantic == class_of_interest])
        )
    return dict(sample_dict)


def get_puma_statistics(classes_of_interest: List = []):
    h5_paths = list(ROOT.glob("*/preprocessed/*.h5"))
    _check_mask = partial(check_mask, classes_of_interest=classes_of_interest)

    with Pool(cpu_count() - 2) as p:
        results = list(
            tqdm(p.imap(_check_mask, h5_paths), desc="Checking PUMA for class distribution", total=len(h5_paths))
        )

    result_dict = defaultdict(list)
    for result in results:
        breakpoint()
        result_dict["filename"].append(result.keys()[0])
    breakpoint()
    df = pd.json_normalize(results, sep="_")
    # breakpoint()
    df.to_csv(ROOT / "summary_puma.csv")


get_puma_statistics([5])
