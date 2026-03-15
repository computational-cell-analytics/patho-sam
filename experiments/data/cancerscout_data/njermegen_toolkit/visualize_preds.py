import os
import napari
import numpy as np
from natsort import natsorted
from glob import glob
import imageio.v3 as imageio


ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/rois_pdl1_ihc"

image_paths = natsorted(glob(os.path.join(ROOT, "images", "*")))
nuclei_preds = natsorted(glob(os.path.join(ROOT, "ignite_output", "*_nuclei.npy")))
pdl1_preds = natsorted(glob(os.path.join(ROOT, "ignite_output", "*_pdl1.npy")))

for img_path, nuclei_pred, pdl1_pred in zip(image_paths, nuclei_preds, pdl1_preds):
    viewer = napari.Viewer()
    img = imageio.imread(img_path)
    viewer.add_image(img, name=os.path.basename(img_path))

    nuclei = np.load(nuclei_pred)
    ncl_coords = nuclei[:, [1, 0]]
    ncl_properties = {"class": nuclei[:, 2]}
    viewer.add_points(
        ncl_coords,
        properties=ncl_properties,
        size=5,
        name="nuclei",
        face_color='class'
    )

    pdl1 = np.load(pdl1_pred)
    pdl1_coords = pdl1[:, [1, 0]] 
    pdl1_properties = {"class": pdl1[:, 2]}
    viewer.add_points(
        pdl1_coords,
        properties=pdl1_properties,
        size=5,
        name="pdl1",
        face_color='class'
    )
    napari.run()
