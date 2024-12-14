import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import imageio
from scipy.io import loadmat


def mat_to_tiff(path):
    label_mat_paths = [p for p in natsorted(glob(os.path.join(path, "*.mat")))]
    label_paths = []
    for mpath in tqdm(label_mat_paths, desc="Preprocessing labels"):
        label_path = mpath.replace(".mat", "_instance_labels.tiff")
        label_paths.append(label_path)
        if os.path.exists(label_path):
            continue
        label = loadmat(mpath)["inst_map"]
        imageio.imwrite(label_path, label)
        os.remove(mpath)


mat_to_tiff('/mnt/lustre-grete/usr/u12649/scratch/models/hovernet/inference/cpm15/pannuke/mat')
