import h5py
import imageio.v3 as imageio

LABEL_V2 = "/mnt/vast-nhr/home/titus.griebel/u23324/label/committed_objects_00101_v2.tiff"
H5_PATH = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/new_tumor_data/new_h5_files/A2020-001011_1-1-1_HE-2021-10-08T10-49-32.h5"
v2_label = imageio.imread(LABEL_V2)
with h5py.File(H5_PATH, 'a') as f:
    f.create_dataset(name="inst_labels/v_2", data=v2_label, compression="gzip")