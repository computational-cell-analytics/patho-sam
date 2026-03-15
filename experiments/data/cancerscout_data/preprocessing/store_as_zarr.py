import zarr
import os
import imageio.v3 as imageio
from glob import glob

ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data"

rois_roots = glob(os.path.join(ROOT, "*_models", "rois*"))