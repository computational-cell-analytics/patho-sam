import os
import imageio.v3 as imageio
from tqdm import tqdm
import h5py
from glob import glob
from patho_sam.postprocessing import remove_disconnected_components_and_fill_holes


ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/new_tumor_data/h5_files"
i = 0
for file_path in glob(os.path.join(ROOT, "*33-49.h5")):
    with h5py.File(file_path, 'a') as f:
        for label_type in ['inst_label', 'inst_pred']:
            if label_type in f:
                raw_label = f[label_type][:]
                processed_label = remove_disconnected_components_and_fill_holes(raw_label)
                del f[label_type]
                f.create_dataset(name=label_type, data=processed_label, compression='gzip')
                i += 1
            else:
                print(f"{label_type} not in {file_path}")


print(f"{i} images processed!")
