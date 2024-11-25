import os
import random
import shutil

def create_val_split(directory, percentage, organ_type=None):
    if organ_type is not None:    
        
        labels_path = os.path.join(directory,organ_type,'labels')
        images_path = os.path.join(directory,organ_type,'images')
        
        label_paths = os.listdir(labels_path)
        image_paths = os.listdir(images_path)
        
        image_paths.sort()
        label_paths.sort()
        val_label_path = os.path.join(directory,organ_type,'val_labels')
        val_image_path = os.path.join(directory,organ_type,'val_images')
    else:
        labels_path = os.path.join(directory,'complete_dataset','labels')
        images_path = os.path.join(directory,'complete_dataset','images')
        
        label_paths = os.listdir(labels_path)
        image_paths = os.listdir(images_path)
        image_paths.sort()
        label_paths.sort()

        val_label_path = os.path.join(directory,'complete_dataset','val_labels')
        val_image_path = os.path.join(directory,'complete_dataset','val_images')
    
    assert os.listdir(val_image_path) == [] or os.path.exists(val_image_path) == False, 'Validation split already exists'
    print('No pre-existing validation set was found. A validation set will be created.')

    if not os.path.exists(val_label_path):
            os.mkdir(val_label_path)
    if not os.path.exists(val_image_path):
            os.mkdir(val_image_path)
    val_count = round(len(image_paths)*percentage)
    print(f'The validation set will consist of {val_count} images.')
    val_indices = random.sample(range(0, (len(image_paths))), val_count)
    val_indices.sort()
    for item in val_indices:
        # print(f'Image {image_paths[item]} and label {label_paths[item]} will be moved to val split')
        image_path = os.path.join(images_path,f'{image_paths[item]}')
        image_destination = os.path.join(val_image_path,f'{image_paths[item]}')
        label_path = os.path.join(labels_path,f'{label_paths[item]}')
        label_destination = os.path.join(val_label_path,f'{label_paths[item]}')
        #print(f'Image origin: {image_path}, image destination: {image_destination}')
    
        shutil.move(image_path, image_destination)
        shutil.move(label_path, label_destination)

directory = '/scratch/users/u11644/data/monusac/loaded_data/'
percentage = 0.05
organ_type = 'prostate'
create_val_split(directory, percentage, organ_type)

