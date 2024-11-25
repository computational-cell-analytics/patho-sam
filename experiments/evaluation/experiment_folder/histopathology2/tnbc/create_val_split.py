import os
import random
import shutil
from natsort import natsorted
from glob import glob
import numpy as np
from skimage.io import imread
import tifffile

def create_val_split(directory, val_percentage, test_percentage, custom_name=None, organ_type=None, split=None, random_seed=42):
    if split is None:
        path = os.path.join(directory, 'complete_dataset')
    else:
        path = os.path.join(directory, split)
    if custom_name is not None:
        labels_src_path = os.path.join(path, 'labels')
        images_src_path = os.path.join(path, 'images')
        label_list = natsorted(glob(os.path.join(labels_src_path, '*.tiff')))
        invalid_image_list = natsorted(glob(os.path.join(images_src_path, '*.tiff')))
        assert len(label_list) == len(invalid_image_list)
        valid_count = 0
        image_list = []
        for image in invalid_image_list:
            data = tifffile.imread(image)
            print(data.shape)
            if data.shape[2] == 3:
                image_list.append(image)
                valid_count += 1
        print(f'{valid_count} from {len(invalid_image_list)} images were valid and did not have shape[2] != 3.')

        # label_src_paths = os.listdir(labels_src_path)
        # image_src_paths = os.listdir(images_src_path)
        # image_src_paths.sort()
        # label_src_paths.sort()

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
    # print(len(image_list))
    # print(image_list[-6])
    # print(val_images)
    for val in val_images:
        # print(f'Image {image_paths[item]} and label {label_paths[item]} will be moved to val split')
        #print(val)
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
            # print(f'Image {image_paths[item]} and label {label_paths[item]} will be moved to val split')
            image_path = test
            image_destination = os.path.join(test_image_dst)
            label_path = os.path.join(labels_src_path, os.path.basename(test))
            label_destination = os.path.join(test_label_dst)
            image_list.remove(test)
            label_list.remove((os.path.join(labels_src_path, (os.path.basename(test)))))
            shutil.copy(image_path, image_destination)
            shutil.copy(label_path, label_destination)
        assert len(os.listdir(os.path.join(test_label_dst))) == len(os.listdir(os.path.join(test_image_dst))), 'label / image count mismatch in test set'

    for train in image_list:
        image_path = train
        image_destination = os.path.join(train_image_dst)
        label_path = os.path.join(labels_src_path, os.path.basename(train))
        label_destination = os.path.join(train_label_dst)
        shutil.copy(image_path, image_destination)
        shutil.copy(label_path, label_destination)
    assert len(os.listdir(os.path.join(train_label_dst))) == len(os.listdir(os.path.join(train_image_dst))), 'label / image count mismatch in train set'
    print(f'Train set: {len(os.listdir(os.path.join(train_image_dst)))} images;  val set: {len(os.listdir(os.path.join(val_image_dst)))} images; test set: {len(os.listdir(os.path.join(test_image_dst)))}')

directory = '/mnt/lustre-grete/usr/u12649/scratch/data/tnbc/loaded_dataset'
val_percentage = 0.05
test_percentage = 0.95
# organ_type = 
#create_val_split(directory, val_percentage, test_percentage, custom_name='test3')

def check_lynsec(path):
    labels_dir = os.path.join(path, 'test_images')
    for image in os.listdir(labels_dir):
        image_path = os.path.join(labels_dir, image)
        npdata = tifffile.imread(image_path)
        print(npdata.shape)

check_lynsec('/mnt/lustre-grete/usr/u12649/scratch/data/tnbc/loaded_dataset/complete_dataset/test2')