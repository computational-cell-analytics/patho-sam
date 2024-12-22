import os
import zarr
import imageio
import numpy as np
from glob import glob
from natsort import natsorted
import zipfile
import shutil

def unzip(path):
    

def zarr_to_tiff(path):
    zarr_dirs = natsorted(glob(os.path.join(path)))