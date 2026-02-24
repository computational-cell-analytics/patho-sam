import os
from glob import glob
import imageio.v3 as imageio
from elf.evaluation import mean_segmentation_accuracy
import numpy as np
from tqdm import tqdm
ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/rois_new_tumor/"

GT_DIR = os.path.join(ROOT, "annotations")
PRED_DIR = os.path.join(ROOT, "segmentations")

msa_list = []

for segmentation_path in tqdm(glob(os.path.join(GT_DIR, "*_label.tiff"))):
    image_name = os.path.basename(segmentation_path)
    gt_path = os.path.join(PRED_DIR, image_name.replace("_label.tiff", ".tiff"))
    # breakpoint()

    if not os.path.exists(gt_path):
        continue
    pred = imageio.imread(segmentation_path)
    gt = imageio.imread(gt_path)
    msa = mean_segmentation_accuracy(pred, gt)
    print(msa)
    msa_list.append(msa)
    print(len(np.unique(gt)))
print(f"mSA over {len(msa_list)} samples: {np.mean(msa_list).round(4)}")



