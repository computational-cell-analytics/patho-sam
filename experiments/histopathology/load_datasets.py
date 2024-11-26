import os
import tifffile
import numpy
from dataloaders import get_dataloaders

def load_datasets(path, datasets=['cpm15', 'cpm17', 'cryonuseg', 'janowczyk', 'lizard', 'lynsec', 'monusac', 'monuseg', 'nuinsseg', 'pannuke', 'puma', 'tnbc']):
    patch_shape = (1, 512, 512)
    for dataset in sorted(datasets):
        if os.path.exists(os.path.join(path, dataset, 'loaded_dataset')):
            continue
        counter = 1
        print(dataset)
        if dataset in ['cpm15','cpm17']:
            assert os.path.exists(os.path.join(path, dataset, dataset)), 'Missing data! For the cpm15 and cpm17 datasets, data has to be downloaded in advance in the format path/cpm15/cpm15/Images and path/cpm17/cpm17/train/Images'
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
                tifffile.imwrite(os.path.join(image_output_path, f'{counter:04}.tiff'), tp_img)
                tifffile.imwrite(os.path.join(label_output_path, f'{counter:04}.tiff'), label_data)
                counter += 1


load_datasets('/mnt/lustre-grete/usr/u12649/scratch/data/test')
