import os
import tifffile as tiff
from dataloaders import get_dataloaders
from tqdm import tqdm
from shutil import rmtree
from util import dataloading_args

DATASETS = ['cpm15', 'cpm17', 'cryonuseg', 'janowczyk', 'lizard', 'lynsec',
            'monusac', 'monuseg', 'nuinsseg', 'pannuke', 'puma', 'tnbc']


def load_datasets(path, datasets=DATASETS, patch_shape=(512, 512)):
    for dataset in tqdm(datasets):
        if os.path.exists(os.path.join(path, dataset, 'loaded_dataset')):
            print(f'{dataset} dataset is loaded already.')
            continue
        counter = 1
        if dataset in ['cpm15', 'cpm17']:
            if not os.path.exists(os.path.join(path, dataset, dataset)):
                print(f'Missing data! For the {dataset} dataset, data has to be downloaded in advance in the format path/{dataset}/{dataset}/Images \n'
                      f'Dataset loading will skip {dataset} and continue with the remaining datasets.')
                continue
        print(f'Loading {dataset} dataset...')
        dpath = os.path.join(path, dataset)
        os.makedirs(dpath, exist_ok=True)
        if dataset not in ['lizard', 'monusac', 'monuseg', 'pannuke']:
            loaders = [get_dataloaders(patch_shape, dpath, dataset)]
        elif dataset == 'lizard':
            loaders = []
            for split in ['split1', 'split2', 'split3']:  # this represents all available splits
                loaders.append(get_dataloaders(patch_shape, dpath, dataset, split))
        elif dataset == 'monusac':
            loaders = []
            for split in ['train', 'test']:  # this represents all available splits
                loaders.append(get_dataloaders(patch_shape, dpath, dataset, split=split))
        elif dataset == 'monuseg':
            loaders = []
            for split in ['train', 'test']:  # this represents all available splits
                loaders.append(get_dataloaders(patch_shape, dpath, dataset, split=split))
        elif dataset == 'pannuke':
            loaders = []
            folds = ['fold_3']  # this represents only fold3 for testing the model; other available folds: fold_1, fold_2
            loaders.append(get_dataloaders(patch_shape, dpath, dataset, split=folds))
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
                if tp_img.shape != (patch_shape[0], patch_shape[1], 3):  # 3 tnbc images had a shape of (512, 512, 2) and had to be sorted out
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
    args = dataloading_args()
    if args.path is not None:
        data_path = args.path
    else:
        data_path = '/mnt/lustre-grete/usr/u12649/scratch/data/test'
    if args.datasets is not None:
        load_datasets(data_path, [args.datasets], args.patch_shape)
    else:
        load_datasets(data_path, patch_shape=args.patch_shape)


if __name__ == "__main__":
    main()
