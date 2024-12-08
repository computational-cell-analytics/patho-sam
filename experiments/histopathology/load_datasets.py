"""
This loads the selected datasets as .tiff files with an image shape of (512, 512, 3) and a label shape of (512, 512).
Alpha channels are deleted and shape deviations excluded.
"""

import os
from tqdm import tqdm
from shutil import rmtree

import tifffile as tiff

from dataloaders import get_dataloaders


def load_datasets(path, datasets=['cpm15', 'cpm17', 'cryonuseg', 'janowczyk', 'lizard', 'lynsec', 'monusac', 'monuseg', 'nuinsseg', 'pannuke', 'puma', 'tnbc']):
    patch_shape = (1, 512, 512) # I will change this to a more cli-friendly structure so patch shape, path and dataset choice can be modified without touching the code
    for dataset in tqdm(sorted(datasets)):
        if os.path.exists(os.path.join(path, dataset, 'loaded_dataset')):
            continue

        counter = 1
        if dataset in ['cpm15', 'cpm17']:
            if not os.path.exists(os.path.join(path, dataset, dataset)):
                print(f'Missing data! For the {dataset} dataset, data has to be downloaded in advance in the format path/{dataset}/{dataset}/Images \n'
                      f'Dataset loading will skip {dataset} and continue with the remaining datasets.')
                continue

        print(f'Loading {dataset} dataset...')
        os.makedirs(os.path.join(path, dataset), exist_ok=True)
        if dataset not in ['cryonuseg', 'lizard', 'monusac', 'monuseg', 'pannuke']:
            loaders = [get_dataloaders(patch_shape, data_path=os.path.join(path, dataset), dataset=dataset)]
        elif dataset == 'cryonuseg':
            loaders = []
            for rater in ['b1', 'b2', 'b3']:  # this represents all available raters
                loaders.append(get_dataloaders(patch_shape, data_path=os.path.join(path, dataset), dataset=dataset, split=rater))
        elif dataset == 'lizard':
            loaders = []
            for split in ['split1', 'split2', 'split3']:  # this represents all available splits
                loaders.append(get_dataloaders(patch_shape, data_path=os.path.join(path, dataset), dataset=dataset, split=split))
        elif dataset == 'monusac':
            loaders = []
            for split in ['train', 'test']:  # this represents all available splits
                loaders.append(get_dataloaders(patch_shape, data_path=os.path.join(path, dataset), dataset=dataset, split=split))
        elif dataset == 'monuseg':
            loaders = []
            for split in ['train', 'test']:  # this represents all available splits
                loaders.append(get_dataloaders(patch_shape, data_path=os.path.join(path, dataset), dataset=dataset, split=split))
        elif dataset == 'pannuke':
            loaders = []
            folds = ['fold_3']  # this represents only fold3 for testing the model; other available folds: fold_1, fold_2
            loaders.append(get_dataloaders(patch_shape, data_path=os.path.join(path, dataset), dataset=dataset, split=folds))

        image_output_path = os.path.join(path, dataset, 'loaded_dataset/complete_dataset/images')
        label_output_path = os.path.join(path, dataset, 'loaded_dataset/complete_dataset/labels')
        os.makedirs(image_output_path, exist_ok=True)
        os.makedirs(label_output_path, exist_ok=True)
        for loader in loaders:
            for image, label in loader:
                image_array = image.numpy()
                label_array = label.numpy()
                squeezed_image = image_array.squeeze()
                label_data = label_array.squeeze()
                tp_img = squeezed_image.transpose(1, 2, 0)
                if tp_img.shape[-1] == 4:  # deletes alpha channel if one exists
                    tp_img = tp_img[..., :-1]
                if tp_img.shape != (512, 512, 3):  # 3 tnbc images had a shape of (512, 512, 2) and had to be sorted out
                    print(f'Incorrect image shape of {tp_img.shape} in {os.path.join(image_output_path, f'{counter:04}.tiff')}')
                    counter += 1
                    continue

                tiff.imwrite(os.path.join(image_output_path, f'{counter:04}.tiff'), tp_img)
                tiff.imwrite(os.path.join(label_output_path, f'{counter:04}.tiff'), label_data)
                counter += 1

        print(f'{dataset} dataset has successfully been loaded.')

    for dataset in datasets:
        if os.path.exists(os.path.join(path, dataset, 'loaded_dataset')):
            for entity in os.listdir(os.path.join(path, dataset)):
                entity_path = os.path.join(path, dataset, entity)
                if entity != 'loaded_dataset' and os.path.isdir(entity_path):
                    rmtree(entity_path)
                elif entity != 'loaded_dataset' and not os.path.isdir(entity_path):
                    os.remove(entity_path)


def main():
    load_datasets('/mnt/lustre-grete/usr/u12649/scratch/data/test')


if __name__ == "__main__":
    main()
