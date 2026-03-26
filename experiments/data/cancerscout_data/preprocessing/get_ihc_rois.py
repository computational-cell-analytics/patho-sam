import numpy as np
import imageio.v3 as imageio
import pyvips
import json
import os


ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/"
JSON_PATH = "/user/titus.griebel/u23324/patho-sam/experiments/data/cancerscout_data/training_rois.json"
SQUARE_LENGTH = 5120

with open(JSON_PATH, 'r') as f:
    data = json.load(f)
split = "train"
img_type = "pdl1_ihc"
rois_dir = os.path.join(ROOT, f"{split}_models", f"rois_{img_type}")
roi_images_dir = os.path.join(rois_dir, "images")
rois_embeddings_dir = os.path.join(rois_dir, "embeddings")
segmentation_dir = os.path.join(rois_dir, "segmentations")
os.makedirs(rois_embeddings_dir, exist_ok=True)
os.makedirs(roi_images_dir, exist_ok=True)
os.makedirs(segmentation_dir, exist_ok=True)
for img_name, roi_position in data['pdl1_ihc'].items():
    non_pyramid_path = os.path.join(ROOT, "train_models", "CancerScout_Lung", "pdl1_ihc", img_name.replace("_pyramid.tiff", ".tiff"))
    roi_name = os.path.basename(non_pyramid_path.split(".")[0])
    output_path = os.path.join(roi_images_dir, f"roi_{roi_name}.tiff")
    if os.path.exists(output_path):
        continue
    image = pyvips.Image.new_from_file(non_pyramid_path, access='sequential')
    patch = image.crop(roi_position[0], roi_position[1], SQUARE_LENGTH, SQUARE_LENGTH)
    patch_np = np.ndarray(buffer=patch.write_to_memory(),
                          dtype=np.uint8,
                          shape=[patch.height, patch.width, patch.bands])
    imageio.imwrite(output_path, patch_np,  plugin="tifffile", compression="zlib")