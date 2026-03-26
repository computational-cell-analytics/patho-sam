import os 
import zarr
from glob import glob

EMBEDDINGS_DIR = '/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/rois_new_tumor/embeddings'
EMBEDDINGS_DIR_ZARR2 = '/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/rois_new_tumor/embeddings_z2'




def copy_group(src_group, dst_group):
    # copy attributes
    try:
        dst_group.attrs.update(dict(src_group.attrs))
    except Exception:
        pass

    for key in src_group.keys():
        obj = src_group[key]

        if isinstance(obj, zarr.Array):
            dst_arr = dst_group.create_array(
                key,
                data=obj[:],        # pass the full array
                chunks=obj.chunks,  # optional, keeps original chunking
                # dtype removed!
            )
            try:
                dst_arr.attrs.update(dict(obj.attrs))
            except Exception:
                pass

        elif isinstance(obj, zarr.Group):
            # recurse into subgroup
            new_group = dst_group.create_group(key)
            copy_group(obj, new_group)


for zarr_file in glob(os.path.join(EMBEDDINGS_DIR, "*")):
    src = zarr.open(zarr_file, mode="r")
    dst = zarr.open(os.path.join(EMBEDDINGS_DIR_ZARR2, os.path.basename(zarr_file)), mode="w")
    dst.attrs.update(dict(src.attrs))
    copy_group(src, dst)

