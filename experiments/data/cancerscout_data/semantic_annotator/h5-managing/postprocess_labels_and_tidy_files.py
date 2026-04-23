from pathlib import Path
import h5py
import imageio.v3 as imageio
from patho_sam.annotation import postprocess_instance_mask

ROOT = ""

for h5_file in Path(ROOT).glob("*.h5"):
    with h5py.File(h5_file, 'a') as f:
        if "object_features" in f.keys():
            del f['object_features']
            del f['seg_ids']
        for inst_label_version in list(f['inst_labels'].keys()):
            inst_label_ds = f[f'inst_labels/{inst_label_version}']
            inst_label = inst_label_ds[:]
            postprocessed_inst_label = postprocess_instance_mask(inst_label)
            f.create_dataset(name=f'inst_labels/{inst_label_version}_tmp', data=postprocessed_inst_label, compression='gzip')
            del f[inst_label_ds]
            f.move(f'inst_labels/{inst_label_version}_tmp', inst_label_ds)