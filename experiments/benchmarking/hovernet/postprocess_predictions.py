import os
import shutil
from glob import glob

import imageio
from natsort import natsorted
from scipy.io import loadmat
from tqdm import tqdm


def mat_to_tiff(path):
    label_mat_paths = [p for p in natsorted(glob(os.path.join(path, "*.mat")))]
    label_paths = []
    for mpath in tqdm(label_mat_paths, desc="Converting labels to .tiff"):
        label_path = mpath.replace(".mat", "_instance_labels.tiff")
        label_paths.append(label_path)
        if os.path.exists(label_path):
            continue
        label = loadmat(mpath)["inst_map"]
        imageio.imwrite(label_path, label)
        os.remove(mpath)


def postprocess_hovernet(output_dir):
    for dataset in [
        "cpm15",
        "cpm17",
        "cryonuseg",
        "janowczyk",
        "lizard",
        "lynsec",
        "monusac",
        "monuseg",
        "nuinsseg",
        "pannuke",
        "puma",
        "tnbc",
    ]:
        for model in ["consep", "cpm17", "kumar", "pannuke", "monusac"]:
            output_path = os.path.join(output_dir, dataset, model)
            mat_to_tiff(os.path.join(output_path, "mat"))
            shutil.rmtree(os.path.join(output_path, "json"))
            shutil.rmtree(os.path.join(output_path, "overlay"))
            if len(os.listdir(os.path.join(output_path, "mat"))) == 0:
                shutil.rmtree(os.path.join(output_path, "mat"))


postprocess_hovernet("/mnt/lustre-grete/usr/u12649/models/hovernet/inference")
