import os 
from glob import glob
from natsort import natsorted
import pandas as pd 

ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/eval_models/CancerScout_Lung/new_tumor"
file_size_dict = {
    "wsi_name": [],
    "wsi_size": [],
}
for idx, file in enumerate(natsorted(glob(os.path.join(ROOT, "*.tiff")))):
    print(os.path.basename(file).split("_")[0], os.path.getsize(file) / 1e9)
    if idx == 10:
        break
