import os
import tifffile as tiff
from dataloaders import get_dataloaders
from tqdm import tqdm
from shutil import rmtree
from util import dataloading_args
import os
from torch_em.data import datasets, MinInstanceSampler, ConcatDataset
import torch.utils.data as data_util
import micro_sam.training as sam_training
import torch_em
import torch
import torch.utils.data as data_util






DATASETS = ['cpm15', 'cpm17', 'janowczyk', 'lizard',
            'monuseg', 'pannuke', 'puma', 'tnbc']


def _get_train_val_split(ds, val_fraction: float = 0.2, test_exists=True):
    if not test_exists:
        ds, _ = _get_train_test_split(ds=ds)
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = data_util.random_split(ds, [1 - val_fraction, val_fraction], generator=generator)
    return train_ds, val_ds, _

def _get_train_test_split(ds, test_fraction: float = 0.2):
    generator = torch.Generator().manual_seed(42)
    train_split, test_split = data_util.random_split(ds, [1 - test_fraction, test_fraction], generator=generator)
    return train_split, test_split

def load_datasets(path, dsets=DATASETS, patch_shape=(512, 512)):
    
    label_dtype = torch.int64
    sampler = MinInstanceSampler(min_num_instances=2)
    raw_transform = sam_training.identity

    cpm15_ds = datasets.get_cpm_dataset(
        path=os.path.join(path, "cpm15"), patch_shape=patch_shape, sampler=sampler, label_dtype=label_dtype,
        raw_transform=raw_transform, data_choice='cpm15'
    )


    cpm17_ds = datasets.get_cpm_dataset(
       path=os.path.join(path, "cpm17"), patch_shape=patch_shape, sampler=sampler, label_dtype=label_dtype,
       raw_transform=raw_transform, data_choice='cpm17'
    )


    janowczyk_ds = datasets.get_janowczyk_dataset(
        path=os.path.join(path, "janowczyk"), patch_shape=patch_shape, sampler=sampler, download=True, 
        label_dtype=label_dtype, raw_transform=raw_transform, annotation="nuclei"
    )


    puma_ds = datasets.get_puma_dataset(
        path=os.path.join(path, "puma"), patch_shape=patch_shape, download=True, sampler=sampler, 
        raw_transform=raw_transform, label_dtype=label_dtype
    )


    tnbc_ds = datasets.get_tnbc_dataset(
        path=os.path.join(path, "tnbc"), patch_shape=patch_shape, download=True, sampler=sampler, 
        label_dtype=label_dtype, ndim=2, raw_transform=raw_transform
    )
    for dataset in tqdm(dsets):
        if os.path.exists(os.path.join(path, dataset, 'loaded_dataset')):
            print(f'{dataset} dataset is loaded already.')
            continue
        counter = 1
        print(f'Loading {dataset} dataset...')
        dpath = os.path.join(path, dataset)
        os.makedirs(dpath, exist_ok=True)
        
        
        if dataset == 'cpm15':
            _, __, test_set = _get_train_val_split(cpm15_ds, test_exists=False)
            test_loader = torch_em.get_data_loader(test_set, batch_size=1, shuffle=True, num_workers=16)
        elif dataset == 'cpm17':
            _, __, test_set = _get_train_val_split(cpm17_ds, test_exists=False)
            test_loader = torch_em.get_data_loader(test_set, batch_size=1, shuffle=True, num_workers=16)
        elif dataset == 'janowczyk':
            _, __, test_set = _get_train_val_split(janowczyk_ds, test_exists=False)
            test_loader = torch_em.get_data_loader(test_set, batch_size=1, shuffle=True, num_workers=16)
        elif dataset == 'lizard':
            for split in ['test']:  
                test_loader = get_dataloaders(patch_shape, dpath, dataset, split)
        elif dataset == 'monuseg':
            for split in ['test']:
                test_loader = get_dataloaders(patch_shape, dpath, dataset, split=split)
        elif dataset == 'pannuke':
            folds = ['fold_3']  # this represents only fold3 for testing the model; other available folds: fold_1, fold_2
            test_loader = get_dataloaders(patch_shape, dpath, dataset, split=folds)
        elif dataset == 'puma':
            _, __, test_set = _get_train_val_split(puma_ds, test_exists=False)
            test_loader = torch_em.get_data_loader(test_set, batch_size=1, shuffle=True, num_workers=16)
        elif dataset == 'tnbc':
            _, __, test_set = _get_train_val_split(tnbc_ds, test_exists=False)
            test_loader = torch_em.get_data_loader(test_set, batch_size=1, shuffle=True, num_workers=16)
        image_output_path = os.path.join(path, dataset, 'loaded_dataset/complete_dataset/images')
        label_output_path = os.path.join(path, dataset, 'loaded_dataset/complete_dataset/labels')
        os.makedirs(image_output_path, exist_ok=True)
        os.makedirs(label_output_path, exist_ok=True)
        for image, label in test_loader:
            image_array = image.numpy()
            label_array = label.numpy()
            squeezed_image = image_array.squeeze()
            label_data = label_array.squeeze()
            tp_img = squeezed_image.transpose(1, 2, 0)
            if tp_img.shape[-1] == 4:  # deletes alpha channel if one exists
                tp_img = tp_img[..., :-1]
            if tp_img.shape != (patch_shape[0], patch_shape[1], 3):  # 3 tnbc images had a shape of (512, 512, 2) and had to be sorted out
                print(f'Incorrect image shape of {tp_img.shape} in {os.path.join(image_output_path, f'{counter:04}.tiff')}')
                counter += 1
                continue
            tiff.imwrite(os.path.join(image_output_path, f'{counter:04}.tiff'), tp_img)
            tiff.imwrite(os.path.join(label_output_path, f'{counter:04}.tiff'), label_data)
            counter += 1
        print(f'{dataset} testset has successfully been loaded.')


def main():
    args = dataloading_args()
    if args.path is not None:
        data_path = args.path
    else:
        data_path = '/mnt/lustre-grete/usr/u12649/scratch/data/test/'
    if args.datasets is not None:
        load_datasets(data_path, [args.datasets], args.patch_shape)
    else:
        load_datasets(data_path, patch_shape=args.patch_shape)


if __name__ == "__main__":
    main()
