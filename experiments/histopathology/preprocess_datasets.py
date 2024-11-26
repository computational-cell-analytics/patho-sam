import os
import random
import shutil
from natsort import natsorted
from glob import glob
import numpy as np
from skimage.io import imread
import tifffile

""" Script to create:
    - a reproducable validation, training and test set from the root directory 
    - evaluation directories for the segmentation modi: automatic instance segmentation, automatic mask generation, iterative prompting with and without boxes for each given model
     """



def preprocess_datasets(eval_path, data_path, random_seed, test_percentage, val_percentage, custom_name='standard_split', model_names=['pannuke_sam', 'vanilla_sam']):
    datasets = ['cpm15', 'cpm17', 'cryonuseg', 'janowczyk', 'lizard', 'lynsec', 'monusac', 'monuseg', 'nuinsseg', 'pannuke', 'puma', 'tnbc']
    for dataset in datasets:
        create_val_split(os.path.join(data_path, dataset, val_percentage=0.05, test_percentage=0.95, custom_name='standard_split', random_seed=42))



preprocess_datasets('/mnt/lustre-grete/usr/u12649/scratch/models', '/mnt/lustre-grete/usr/u12649/scratch/data/test',)


def create_val_split(directory, val_percentage, test_percentage, custom_name, organ_type=None, split=None, random_seed=42):
    if split is None:
        path = os.path.join(directory, 'loaded_dataset/complete_dataset')
    else:
        path = os.path.join(directory, split)
    labels_src_path = os.path.join(path, 'labels')
    images_src_path = os.path.join(path, 'images')
    label_list = natsorted(glob(os.path.join(labels_src_path, '*.tiff')))
    image_list = natsorted(glob(os.path.join(images_src_path, '*.tiff')))
    assert len(label_list) == len(image_list)
    val_label_dst = os.path.join(path, custom_name, 'val_labels')
    val_image_dst = os.path.join(path, custom_name, 'val_images')
    test_label_dst = os.path.join(path, custom_name, 'test_labels')
    test_image_dst = os.path.join(path, custom_name, 'test_images')
    train_label_dst = os.path.join(path, custom_name, 'train_labels')
    train_image_dst = os.path.join(path, custom_name, 'train_images')
    os.makedirs(val_label_dst, exist_ok=True)
    os.makedirs(val_image_dst, exist_ok=True)
    os.makedirs(test_label_dst, exist_ok=True)
    os.makedirs(test_image_dst, exist_ok=True)
    os.makedirs(train_label_dst, exist_ok=True)
    os.makedirs(train_image_dst, exist_ok=True)
    assert os.listdir(val_image_dst) == [], 'Validation split already exists'
    assert os.listdir(val_label_dst) == [], 'Validation split already exists'
    assert os.listdir(test_image_dst) == [], 'Test split already exists'
    assert os.listdir(test_label_dst) == [], 'Test split already exists'
    assert os.listdir(train_image_dst) == [], 'Train split already exists'
    assert os.listdir(train_label_dst) == [], 'Train split already exists'
    print('No pre-existing validation or test set was found. A validation set will be created.')
    val_count = round(len(image_list)*val_percentage)
    test_count = round(len(image_list)*test_percentage)
    print(f'The validation set will consist of {val_count} images.')
    print(f'The test set will consist of {test_count} images.')
    random.seed(random_seed)
    val_indices = random.sample(range(0, (len(image_list))), val_count)
    val_images = [image_list[x] for x in val_indices]
    for val in val_images:
        image_path = val
        image_destination = os.path.join(val_image_dst)
        label_path = os.path.join(labels_src_path, os.path.basename(val))
        label_destination = os.path.join(val_label_dst)
        shutil.copy(image_path, image_destination)
        shutil.copy(label_path, label_destination)
        image_list.remove(val)
        label_list.remove((os.path.join(labels_src_path, (os.path.basename(val)))))
    assert len(os.listdir(os.path.join(val_label_dst))) == len(os.listdir(os.path.join(val_image_dst))), 'label / image count mismatch in val set'
    test_indices = random.sample(range(0, (len(image_list))), test_count)
    if test_indices == 0:
        pass
    else:
        test_images = [image_list[x] for x in test_indices]
        test_images.sort(reverse=True)
        for test in test_images:
            image_path = test
            image_destination = os.path.join(test_image_dst)
            label_path = os.path.join(labels_src_path, os.path.basename(test))
            label_destination = os.path.join(test_label_dst)
            image_list.remove(test)
            label_list.remove((os.path.join(labels_src_path, (os.path.basename(test)))))
            shutil.copy(image_path, image_destination)
            shutil.copy(label_path, label_destination)
        assert len(os.listdir(os.path.join(test_label_dst))) == len(os.listdir(os.path.join(test_image_dst))), 'label / image count mismatch in test set'
    #residual images are per default in the train set
    for train in image_list:  
        image_path = train
        image_destination = os.path.join(train_image_dst)
        label_path = os.path.join(labels_src_path, os.path.basename(train))
        label_destination = os.path.join(train_label_dst)
        shutil.copy(image_path, image_destination)
        shutil.copy(label_path, label_destination)
    assert len(os.listdir(os.path.join(train_label_dst))) == len(os.listdir(os.path.join(train_image_dst))), 'label / image count mismatch in train set'
    print(f'Train set: {len(os.listdir(os.path.join(train_image_dst)))} images;  val set: {len(os.listdir(os.path.join(val_image_dst)))} images; test set: {len(os.listdir(os.path.join(test_image_dst)))}')


directory = '/mnt/lustre-grete/usr/u12649/scratch/data/puma/' 
val_percentage = 0.05
test_percentage = 0.95
create_val_split(directory, val_percentage, test_percentage, custom_name='test2')

def create_eval_directories(path):
    datasets = ['cryonuseg', 'lynsec', 'lizard', 'pannuke', 'monusac', 'monuseg', 'tnbc', 'puma', 'jano', 'nuinsseg', 'cpm15', 'cpm17']
    for dataset in datasets:
        dataset_path = os.path.join(path, f'{dataset}_eval')
        for mode in ['instance', 'boxes', 'points', 'amg']:
            os.makedirs(os.path.join(dataset_path, mode), exist_ok=True)


create_eval_directories('/mnt/lustre-grete/usr/u12649/scratch/models/old_pannuke_sam_eval')
