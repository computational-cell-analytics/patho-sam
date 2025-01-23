import os
from glob import glob
from tqdm import tqdm

import cv2
import imageio.v3 as imageio
from natsort import natsorted

import json

import numpy as np


ROOT = "/mnt/lustre-grete/usr/u12649/models/hovernet_types/inference/pannuke/pannuke"


def json_to_tiff():
    label_json_paths = [p for p in natsorted(glob(os.path.join(ROOT, "json", "*.json")))]
    img_shape = (512, 512)
    os.makedirs(os.path.join(ROOT, "semantic"), exist_ok=True)
    for mpath in tqdm(label_json_paths, desc="Postprocessing labels"):
        label_path = os.path.join(ROOT, "semantic", os.path.basename(mpath.replace(".json", ".tiff")))
        with open(mpath, 'r') as file:
            data = json.load(file)
            pred_class_map = np.zeros(img_shape, dtype=np.int32)
            for id, cell_data in enumerate(data['nuc'].items(), start=1):
                cell_data = cell_data[1]
                contour = np.array(cell_data["contour"])
                contour[:, 0] = np.clip(contour[:, 0], 0, img_shape[1])
                contour[:, 1] = np.clip(contour[:, 1], 0, img_shape[0])
                contour = contour.reshape((-1, 1, 2))
                cell_type = cell_data["type"]
                contour = np.vstack((contour, [contour[0]]))
                contour = contour.astype(np.int32)
                cv2.fillPoly(pred_class_map, [contour], cell_type)

        imageio.imwrite(label_path, pred_class_map)
