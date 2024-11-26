import os
from skimage import io
import tifffile 
from glob import glob
from tqdm import tqdm
import numpy as np
from natsort import natsorted

def delete_alpha_channel(path):
    for filename in tqdm(glob(os.path.join(path, '*.tiff'))):
        image_path = filename
        data = io.imread(image_path)
        print(data.shape)
        if data.shape[-1] == 4:
            cleansed_data = data[:,:,:3]
            name, ext = os.path.splitext(filename)
            new_filename = name+'.tiff'
            output_path = os.path.join(path, f'{new_filename}')
            tifffile.imwrite(output_path, cleansed_data)
            print(f'Image {name} was successfully cleansed of its alpha channel')


delete_alpha_channel('/mnt/lustre-grete/usr/u12649/scratch/data/nuinsseg/loaded_dataset/complete_dataset/images')


def check_for_empty_tiff(path):
    empty_count = 0
    file_list = natsorted(glob(os.path.join(path, 'labels','*.tiff')))
    for image_path in file_list:
        photo = io.imread(image_path)
        data = np.array(photo)
        unique_elements = np.unique(data)
        print(np.shape(data))
        if len(unique_elements) == 1:
            print(f'Image {os.path.basename(image_path)} does not contain labels and will be removed.')
            empty_count += 1
            os.remove(image_path)
            os.remove(os.path.join(path, 'images', f'{os.path.basename(image_path)}'))
            assert len(os.listdir(os.path.join(path, 'labels'))) == len(os.listdir(os.path.join(path, 'images')))
    print(f'{empty_count} labels were empty')
    label_len = len(os.listdir(os.path.join(path, 'labels')))
    print(f'There are {label_len} images left')


check_for_empty_tiff('/mnt/lustre-grete/usr/u12649/scratch/data/nuinsseg/loaded_dataset/complete_dataset')