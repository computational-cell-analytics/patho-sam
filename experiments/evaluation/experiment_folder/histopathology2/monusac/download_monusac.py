import torch
import os
from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.histopathology.monusac import get_monusac_loader
from torch_em.transform.label import PerObjectDistanceTransform
import numpy as np
import micro_sam.training as sam_training
import tifffile
import skimage.io


def get_dataloaders(patch_shape, data_path, organ_type):
    """This returns the pannuke data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/histopathology/pannuke.py
    It will automatically download the pannuke data.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=25
    )
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
    sampler = MinInstanceSampler(min_num_instances=3)

    train_loader = get_monusac_loader(
        path=data_path,
        patch_shape=patch_shape,
        split = "train",
        batch_size=1,
        organ_type=organ_type,
        download=True,
        offsets=None,
        boundaries=False,
        binary=False,
        raw_transform=raw_transform,
        sampler=sampler
    )
    val_loader = get_monusac_loader(
        path=data_path,
        patch_shape=patch_shape,
        split = "test",
        batch_size=1,
        organ_type=organ_type,
        download=True,
        offsets=None,
        boundaries=False,
        binary=False,
        raw_transform=raw_transform,
        sampler=sampler
    )
    return train_loader, val_loader

    

def delete_alpha_channel(path):
   for filename in os.listdir(path):
      image_path = os.path.join(path, filename)
      data = skimage.io.imread(image_path)
      print(data.shape)
      if data.shape[-1] == 4:
         cleansed_data = data[:,:,:3]
         output_path = os.path.join(path, f'{filename}')
         tifffile.imwrite(output_path, cleansed_data)
         print(f'Image {(os.listdir(path).index(filename))+1} was successfully cleansed of its alpha channel')




def load_and_save_monusac(directory, organ_type=None):
    data_path = '/mnt/lustre-grete/usr/u12649/scratch/data/monusac/download/complete_dataset'
    os.makedirs(data_path, exist_ok=True)
    if organ_type is not None:
        organ_combination = ''
        for organ in organ_type:
            organ_combination = organ_combination + '_' + organ
        image_output_path = os.path.join(directory, organ_combination, 'images') 
        label_output_path = os.path.join(directory, organ_combination, 'labels') 
    else:
        image_output_path = os.path.join(directory, 'complete_dataset', 'images') 
        label_output_path = os.path.join(directory, 'complete_dataset', 'labels')
    train_loader, val_loader = get_dataloaders(patch_shape=(1, 512, 512), data_path=data_path, organ_type=organ_type)       
    counter = 1
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)
    assert os.listdir(image_output_path) == [], 'Images are loaded already'
    assert os.listdir(label_output_path) == [], 'Labels are loaded already'
    for image,label in train_loader:
       image_array = image.numpy()
       label_array = label.numpy()
       #print(f'Image {counter:04} original shape: {np.shape(image_array)}')
       squeezed_image = image_array.squeeze()
       squeezed_label = label_array.squeeze()
       transposed_image_array = squeezed_image.transpose(1,2,0)
       print(f'Image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(squeezed_label)}')
       tif_image_output_path = os.path.join(image_output_path,f'{counter:04}.tiff')
       tifffile.imwrite(tif_image_output_path, transposed_image_array)
       tif_label_output_path = os.path.join(label_output_path,f'{counter:04}.tiff')
       tifffile.imwrite(tif_label_output_path, squeezed_label)
       counter+=1
    for image, label in val_loader:
      image_array = image.numpy()
      label_array = label.numpy()
      #print(f'Image {counter:04} original shape: {np.shape(image_array)}')
      squeezed_image = image_array.squeeze()
      squeezed_label = label_array.squeeze()
      transposed_image_array = squeezed_image.transpose(1,2,0)
      print(f'image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(squeezed_label)}')
      tif_image_output_path = os.path.join(image_output_path,f'{counter:04}.tiff')
      tifffile.imwrite(tif_image_output_path, transposed_image_array)
      tif_label_output_path = os.path.join(label_output_path,f'{counter:04}.tiff')
      tifffile.imwrite(tif_label_output_path, squeezed_label)
      counter+=1
    #delete_alpha_channel(image_output_path) 
        
       
    
load_and_save_monusac('/mnt/lustre-grete/usr/u12649/scratch/data/monusac/')


# for image, label in train_loader:
#     print(image,label)


# def delete_alpha_channel(path):
#    for filename in os.listdir(path):
#       image_path = os.path.join(path, filename)
#       with tifffile.TiffFile(image_path) as tif:
#          data = np.array(Image.open(image_path))
#          if data.shape[-1] == 4:
#           cleansed_data = data[:,:,:3]
#           output_path = os.path.join(path, f'{filename}')
#           tifffile.imwrite(output_path, cleansed_data)
#           print(f'Image {(os.listdir(path).index(filename))} was successfully cleansed of its alpha channel')


#delete_alpha_channel('/scratch/users/u11644/data/monusac/monusac_test/complete_images')


#OLD PROVEN CODE:
# def load_and_save_monusac(directory, organ_type=None):
#     if organ_type != None:
#        data_path = os.path.join('/scratch/users/u11644/data/monusac/download',f'{organ_type}')
#        if not os.path.exists(data_path):
#           os.makedirs(data_path)
#        image_output_path = os.path.join(directory, organ_type, 'images') 
#        label_output_path = os.path.join(directory, organ_type, 'labels') 
#     else:
#        data_path = '/scratch/users/u11644/data/monusac/download/complete_dataset'
#        if not os.path.exists(data_path):
#           os.makedirs(data_path)
#        image_output_path = os.path.join(directory, 'complete_dataset', 'images') 
#        label_output_path = os.path.join(directory, 'complete_dataset', 'labels')
#     train_loader, val_loader = get_dataloaders(patch_shape=(1,512,512),data_path=data_path, organ_type=organ_type)
#     counter = 1     
#     assert os.listdir(image_output_path) == []
#     assert os.listdir(label_output_path)
#     if not os.path.exists(image_output_path):
#        os.makedirs(image_output_path)
#     if not os.path.exists(label_output_path):
#        os.makedirs(label_output_path)
#     for image,label in train_loader:
#        image_array = image.numpy()
#        label_array = label.numpy()
#        #print(f'Image {counter:04} original shape: {np.shape(image_array)}')
#        squeezed_image = image_array.squeeze()
#        squeezed_label = label_array.squeeze()
#        transposed_image_array = squeezed_image.transpose(1,2,0)
#        print(f'Image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(squeezed_label)}')
#        tif_image_output_path = os.path.join(image_output_path,f'{counter:04}.tiff')
#        tifffile.imwrite(tif_image_output_path, transposed_image_array)
#        tif_label_output_path = os.path.join(label_output_path,f'{counter:04}.tiff')
#        tifffile.imwrite(tif_label_output_path, squeezed_label)
#        counter+=1
#     for image,label in val_loader:
#        image_array = image.numpy()
#        label_array = label.numpy()
#        #print(f'Image {counter:04} original shape: {np.shape(image_array)}')
#        squeezed_image = image_array.squeeze()
#        squeezed_label = label_array.squeeze()
#        transposed_image_array = squeezed_image.transpose(1,2,0)
#        print(f'image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(squeezed_label)}')
#        tif_image_output_path = os.path.join(image_output_path,f'{counter:04}.tiff')
#        tifffile.imwrite(tif_image_output_path, transposed_image_array)
#        tif_label_output_path = os.path.join(label_output_path,f'{counter:04}.tiff')
#        tifffile.imwrite(tif_label_output_path, squeezed_label)
#        counter+=1
   # delete_alpha_channel(image_output_path)#
