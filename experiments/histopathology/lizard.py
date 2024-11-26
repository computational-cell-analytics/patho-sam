import os
import warnings
from glob import glob
from shutil import rmtree
import numpy as np
import h5py
import imageio.v3 as imageio
import torch_em
from natsort import natsorted
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training
from scipy.io import loadmat
from tqdm import tqdm
from torch_em.data.datasets import util
import pandas as pd
import tifffile

# TODO: the links don't work anymore (?)
# workaround to still make this work (kaggle still has the dataset in the same structure):
#   - download the zip files manually from here - https://www.kaggle.com/datasets/aadimator/lizard-dataset
#   - Kaggle API (TODO) - `kaggle datasets download -d aadimator/lizard-dataset`


def _extract_images(image_folder, label_folder, output_dir, split):
    image_files = glob(os.path.join(image_folder, "*.png"))
    split_dict = create_split_dicts('/mnt/lustre-grete/usr/u12649/scratch/data/lizard/lizard_labels/Lizard_Labels/info.csv')
    output_path = os.path.join(output_dir, split)
    os.makedirs(output_path, exist_ok=True)
    for image_file in tqdm(image_files, desc=f"Extract images from {image_folder}"):
        fname = os.path.basename(image_file)
        label_file = os.path.join(label_folder, fname.replace(".png", ".mat"))
        assert os.path.exists(label_file), label_file
        image = imageio.imread(image_file)
        labels = loadmat(label_file) 
        segmentation = labels["inst_map"]
        assert image.shape[:-1] == segmentation.shape
        classes = labels["class"]
        image = image.transpose((2, 0, 1))
        assert image.shape[1:] == segmentation.shape
        name, _ = os.path.splitext(fname)
        output_file = os.path.join(output_path, fname.replace(".png", ".h5"))  
        if name in split_dict[split] and not os.path.exists(output_file):
            with h5py.File(output_file, "a") as f:
                f.create_dataset("image", data=image, compression="gzip")
                f.create_dataset("labels/segmentation", data=segmentation, compression="gzip")
                f.create_dataset("labels/classes", data=classes, compression="gzip")  


def get_tiffs(path, split):
    output_dir = os.path.join(path, split)
    os.makedirs((os.path.join(output_dir, 'images')), exist_ok=True)
    os.makedirs((os.path.join(output_dir, 'labels')), exist_ok=True)
    for file in glob(os.path.join(path, split, '*.h5')): 
        with h5py.File(file, 'r') as f:
            img_data = f['image']
            label_data = f['labels/segmentation']
            basename = os.path.basename(file)
            name, ext = os.path.splitext(basename)
            img_output_path = os.path.join(output_dir, 'images', f'{name}.tiff')
            tifffile.imwrite(img_output_path, img_data)
            label_output_path = os.path.join(output_dir, 'labels', f'{name}.tiff')
            tifffile.imwrite(label_output_path, label_data)


def _require_lizard_data(path, download, split):
    """Download the Lizard dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.
    """
    if not os.path.exists(os.path.join(path, "lizard_images1")):
        util.download_source_kaggle(path=path, dataset_name="aadimator/lizard-dataset", download=download)
        zip_path = os.path.join(path, "lizard-dataset.zip")
        util.unzip(zip_path=zip_path, dst=path)

    image_files = glob(os.path.join(path, "*.h5"))
    if len(image_files) > 0:
        return

    os.makedirs(path, exist_ok=True)

    image_folder1 = os.path.join(path, "lizard_images1", "Lizard_Images1")
    image_folder2 = os.path.join(path, "lizard_images2",  "Lizard_Images2")
    label_folder = os.path.join(path, "lizard_labels", "Lizard_Labels")

    assert os.path.exists(image_folder1), image_folder1
    assert os.path.exists(image_folder2), image_folder2
    assert os.path.exists(label_folder), label_folder

    _extract_images(image_folder1, os.path.join(label_folder, "Labels"), path, split)
    _extract_images(image_folder2, os.path.join(label_folder, "Labels"), path, split)

    rmtree(os.path.join(path, "lizard_images1"))
    rmtree(os.path.join(path, "lizard_images2"))
    rmtree(os.path.join(path, "lizard_labels"))
    rmtree(os.path.join(path, "overlay"))


def get_lizard_dataset(path, patch_shape, split, download=False, **kwargs):
    """Dataset for the segmentation of nuclei in histopathology.

    This dataset is from the publication https://doi.org/10.48550/arXiv.2108.11195.
    Please cite it if you use this dataset for a publication.
    """
    if download:
        warnings.warn(
            "The download link does not work right now. "
            "Please manually download the zip files from https://www.kaggle.com/datasets/aadimator/lizard-dataset"
        )

    _require_lizard_data(path, download, split)

    get_tiffs(path, split)
    image_paths = natsorted(glob(os.path.join(path, split, 'images', '*.tiff')))
    label_paths = natsorted(glob(os.path.join(path, split, 'labels', '*.tiff')))    
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=False, boundaries=False, offsets=None
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


# TODO implement loading the classification labels
# TODO implement selecting different tissue types
# TODO implement train / val / test split (is pre-defined in a csv) --> done


def get_lizard_loader(path, patch_shape, batch_size, split, download=False, **kwargs):
    """Dataloader for the segmentation of nuclei in histopathology. See 'get_lizard_dataset' for details."""
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_lizard_dataset(path, patch_shape, split, download=download, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)



def create_split_dicts(csv_path):
    df = pd.read_csv(csv_path)
    split1_list = []
    split2_list = []
    split3_list = []
    for i in df.index:
        split = df['Split'].iloc[i]
        if split == 1:
            split1_list.append(df['Filename'].iloc[i])
        elif split == 2:
            split2_list.append(df['Filename'].iloc[i])
        elif split == 3:
            split3_list.append(df['Filename'].iloc[i])
    split_dict = {'split1':split1_list, 'split2':split2_list, 'split3':split3_list}
    return split_dict

def get_dataloaders(patch_shape, data_path, split):
    """This returns the pannuke data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/histopathology/pannuke.py
    It will automatically download the pannuke data.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
    sampler = MinInstanceSampler(min_num_instances=3)
    split_loader = get_lizard_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=1,
        split=split,
        download=True,
        raw_transform=raw_transform,
        sampler=sampler,
    )
    return split_loader


if __name__ == "__main__":
    get_lizard_loader()

# def load_lizard_dataset(path):
#     counter = 1
#     _path = os.path.join(path, 'loaded_dataset', 'complete_dataset')
#     for split in ['split1', 'split2', 'split3']:
#         split_loader = get_dataloaders(patch_shape=(1,512,512), data_path=path, split=split)

#         image_output_path = os.path.join(_path, 'images')
#         label_output_path = os.path.join(_path, 'labels') 
#         os.makedirs(image_output_path, exist_ok=True)
#         os.makedirs(label_output_path, exist_ok=True)
#         for image, label in split_loader:
#             image_array = image.numpy()
#             label_array = label.numpy()
#             squeezed_image = image_array.squeeze()
#             label_data = label_array.squeeze()          
#             transposed_image_array = squeezed_image.transpose(1,2,0)
#             print(f'image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(label_data)}')
#             tif_image_output_path = os.path.join(image_output_path,f'{counter:04}.tiff')
#             tifffile.imwrite(tif_image_output_path, transposed_image_array)
#             tif_label_output_path = os.path.join(label_output_path,f'{counter:04}.tiff')
#             tifffile.imwrite(tif_label_output_path, label_data)
#             counter+=1

# load_lizard_dataset('/mnt/lustre-grete/usr/u12649/scratch/data/test/lizard/test')
