import h5py
from glob import glob
import imageio.v3 as imageio
import os
from natsort import natsorted
import numpy as np


ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data"

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


for roi_dir in glob(os.path.join(ROOT, "*_models", "rois_*")):
    if roi_dir.endswith(".gz"):
        continue
    print(roi_dir)
    h5_dir = os.path.join(roi_dir, "h5_files")
    os.makedirs(h5_dir, exist_ok=True)
    for filename in [os.path.basename(path) for path in glob(os.path.join(roi_dir, "embeddings", "*"))]:

        with h5py.File(os.path.join(h5_dir, f"{filename}.h5"), 'a') as f:
            for data_type in ["sem_label", "inst_label", "img", "inst_pred", "object_features", "seg_ids", "ignite_pred"]:
                if data_type not in f.keys() or data_type == "inst_label":
                    if data_type == "inst_label" and "inst_label" in f.keys():
                        inst_label = f["inst_label"][:]
                        del f["inst_label"]
                        f.require_group("inst_labels")
                        if "inst_labels/v1" in f:
                            del f["inst_labels/v1"]

                        f.create_dataset("inst_labels/v1", data=inst_label)
                        print(f["inst_labels"]["v1"])
                    else:
                        data = load_data(natsorted(glob(os.path.join(roi_dir, data_dict[data_type], f"*{filename}*"))))
                        if data is not None:
                            f.create_dataset(data_type, data=data)
    
            # print(f.keys())