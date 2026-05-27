import os

import imageio.v3 as imageio
import numpy as np
from elf.evaluation import mean_segmentation_accuracy, precision, recall, f1
from tqdm import tqdm

ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data"


# TODO
# implement evaluation with recall, precision, f1 score, msa and store everything in a large dataframe
# along with the wsi name

msa_list = []

nuclei_count = []


if not os.path.exists(gt_path):
    continue
pred = imageio.imread(pred_path)
gt = imageio.imread(gt_path)
msa = mean_segmentation_accuracy(pred, gt)
print(msa)
msa_list.append(msa)
nuclei_count_image = len(np.unique(gt))
print(f"{nuclei_count_image} nuclei in image")
nuclei_count.append(nuclei_count_image)

print(f"total: {np.sum(nuclei_count)} \n mean: {np.mean(nuclei_count)}, \n std dev: {np.std(nuclei_count)}")
print(f"mSA over {len(msa_list)} samples: {np.mean(msa_list).round(4)}")
