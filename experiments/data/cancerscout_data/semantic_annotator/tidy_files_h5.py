import h5py
from glob import glob
import imageio.v3 as imageio
import os
from natsort import natsorted
import numpy as np
from pathlib import Path
from tqdm import tqdm


ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/new_tumor_data"

arr = np.zeros((100, 100))

data_dict = {
    "sem_label": "semantic_labels",
    "inst_label": "annotations",
    "img": "images",
    "object_features": "cached_features",
    "seg_ids": "cached_seg_ids",
    "inst_pred": "segmentations",
    "ignite_pred": "ignite_output"
}


def load_data(paths):
    if not paths:
        return None
    else:
        path = paths[0]
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith((".tif", ".tiff")):
        return imageio.imread(path)


def tidy_h5_files(path):
    h5_dir = Path(path) / "new_h5_files"
    for filename in tqdm(list(h5_dir.glob("*.h5"))):
        with h5py.File(filename, 'a') as f:
            for data in ["object_features", "seg_ids"]:
                if data in f:
                    del f[data]
                    print(f"{data} deleted!")
            # v1_label = f['inst_labels/v1'][:]
            # f.create_dataset("inst_labels/v_1", data=v1_label, compression="gzip")
            # del f['inst_labels/v1']



tidy_h5_files(ROOT)