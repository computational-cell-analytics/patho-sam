import os
from torch_em.data.datasets import util
import pandas as pd
import h5py
from glob import glob

def unzip():
    path = '/mnt/lustre-grete/usr/u12649/scratch/data/lizard'
    zip_path = os.path.join('/mnt/lustre-grete/usr/u12649/scratch/data/lizard', "archive.zip")
    #util.download_source(zip_path, LABEL_URL, download=download, checksum=LABEL_CHECKSUM)
    util.unzip(zip_path, path, remove=True)

#unzip()
# csv_path = '/mnt/lustre-grete/usr/u12649/scratch/data/lizard/lizard_labels/Lizard_Labels/info.csv'
# df = pd.read_csv(csv_path)


# split_1 = df[(df['Split'] == 1)]
# split_2 = df[(df['Split'] == 2)]
# split_3 = df[(df['Split'] == 3)]

# print(len(split_1))
# print(len(split_2))
# print(len(split_3))

# split_1.to_csv('split_1_list.csv')

# def iloc_splits(df):
#     split1_list = []
#     split2_list = []
#     split3_list = []
#     for i in df.index:
#         split = df['Split'].iloc[i]
#         if split == 1:
#             split1_list.append(df['Filename'].iloc[i])
#         elif split == 2:
#             split2_list.append(df['Filename'].iloc[i])
#         elif split == 3:
#             split3_list.append(df['Filename'].iloc[i])
#     print(len(split1_list))
#     for i in split3_list:
#         print(i)

# #iloc_splits(df)
def check_dataset_shapes(path):
    for image in glob(os.path.join(path, '*.h5')):
        with h5py.File(image, 'r') as f:
            for name, item in f.items():
                if isinstance(item, h5py.Dataset):
                    shape = item.shape
                    dtype = item.dtype
                    print(f"Dataset: {name}, Shape: {shape}, Data Type: {dtype}")
                    # if len(shape) !=3 and name == 'image'
            # labels = f['labels']
            # print(list(labels.keys()))
            # for name, item in labels.items():
            #     if isinstance(item, h5py.Dataset):
            #         dtype = item.dtype
            #         shape = item.shape
            #         print(f"Dataset: {name}, Shape: {shape}, Data Type: {dtype}")
            #         print(shape)
            #         if len(shape) != 3:
            #             print(f'Image {image} has an unexpected shape of {shape}')

file = '/mnt/lustre-grete/usr/u12649/scratch/data/lizard/split1'
#check_dataset_shapes(file)

raw_key = "image"
label_key = "labels"

# def _load_image_collection_dataset(raw_paths, raw_key, label_paths, label_key, roi, **kwargs):
    
#     def _get_paths(rpath, rkey, lpath, lkey, this_roi=None):
#         rpath = glob(os.path.join(rpath, rkey))
#         rpath.sort()
#         if len(rpath) == 0:
#             raise ValueError(f"Could not find any images for pattern {os.path.join(rpath, rkey)}")
#         lpath = glob(os.path.join(lpath, lkey))
#         lpath.sort()
#         if len(rpath) != len(lpath):
#             raise ValueError(f"Expect same number of raw and label images, got {len(rpath)}, {len(lpath)}")

#         if this_roi is not None:
#             rpath, lpath = rpath[roi], lpath[roi]

#         return rpath, lpath

#     patch_shape = kwargs.pop("patch_shape")
#     if patch_shape is not None:
#         if len(patch_shape) == 3:
#             if patch_shape[0] != 1:
#                 raise ValueError(f"Image collection dataset expects 2d patch shape, got {patch_shape}")
#             patch_shape = patch_shape[1:]
#         assert len(patch_shape) == 2

#     if isinstance(raw_paths, str):
#         raw_paths, label_paths = _get_paths(raw_paths, raw_key, label_paths, label_key, roi)
#         ds = ImageCollectionDataset(raw_paths, label_paths, patch_shape=patch_shape, **kwargs)
#     elif raw_key is None:
#         assert label_key is None
#         assert isinstance(raw_paths, (list, tuple)) and isinstance(label_paths, (list, tuple))
#         assert len(raw_paths) == len(label_paths)
#         ds = ImageCollectionDataset(raw_paths, label_paths, patch_shape=patch_shape, **kwargs)
#     else:
#         ds = []
#         n_samples = kwargs.pop("n_samples", None)
#         samples_per_ds = (
#             [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples, raw_paths, raw_key)
#         )
#         if roi is None:
#             roi = len(raw_paths) * [None]
#         assert len(roi) == len(raw_paths)
#         for i, (raw_path, label_path, this_roi) in enumerate(zip(raw_paths, label_paths, roi)):
#             rpath, lpath = _get_paths(raw_path, raw_key, label_path, label_key, this_roi)
#             dset = ImageCollectionDataset(rpath, lpath, patch_shape=patch_shape, n_samples=samples_per_ds[i], **kwargs)
#             ds.append(dset)
#         ds = ConcatDataset(*ds)
#     return ds

def getpaths(rpath, rkey):
    rpath = glob(os.path.join(rpath, rkey))
    rpath.sort()
    print(len(rpath))

path = '/mnt/lustre-grete/usr/u12649/scratch/data/lizard/'
split = 'split1'
data_paths = glob(os.path.join(path, split, "*.h5"))

getpaths(data_paths,'image')
