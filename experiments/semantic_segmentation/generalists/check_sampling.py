import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from functools import partial
from torch_em.data.datasets.histopathology.pannuke import get_pannuke_dataset
import micro_sam.training as sam_training
import hashlib
from torch.utils.data import RandomSampler
from patho_sam.training.util import get_sampler, build_transforms, geometric_transforms, photometric_transforms
from torch_em.data import MinTwoInstanceSampler
import imageio.v3 as imageio


PATH = os.environ.get("WORK")
data_path = os.path.join('/mnt/lustre-grete/usr/u12649/data/test', 'pannuke')

result_dir = "/mnt/lustre-grete/usr/u12649/data/test/sampling_results"

patch_shape = (1, 256, 256)
transforms = True

sampler = None
label_dtype = torch.float32

if transforms:
    geometric_seq, photometric_seq = build_transforms(patch_shape)
    transform = partial(geometric_transforms, seq=geometric_seq)
    raw_transform = partial(photometric_transforms, seq=photometric_seq)
else:
    transform = None
    raw_transform = sam_training.identity


semantic_dataset = get_pannuke_dataset(
    path=data_path,
    patch_shape=patch_shape,
    ndim=2,
    folds=["fold_1", "fold_2"],
    custom_label_choice="semantic",
    sampler=sampler,
    label_dtype=label_dtype,
    raw_transform=raw_transform,
    download=True,
    deterministic_indices=True,
    transform=transform,
)

instance_dataset = get_pannuke_dataset(
    path=data_path,
    patch_shape=patch_shape,
    ndim=2,
    folds=["fold_1", "fold_2"],
    custom_label_choice="instances",
    sampler=sampler,
    label_dtype=label_dtype,
    raw_transform=raw_transform,
    download=True,
    deterministic_indices=True,
    transform=transform
)

random_inst_dataset = get_pannuke_dataset(
    path=data_path,
    patch_shape=patch_shape,
    ndim=2,
    folds=["fold_1", "fold_2"],
    custom_label_choice="instances",
    sampler=MinTwoInstanceSampler(),
    label_dtype=label_dtype,
    raw_transform=raw_transform,
    download=True,
    transform=transform,
)

random_sem_dataset = get_pannuke_dataset(
    path=data_path,
    patch_shape=patch_shape,
    ndim=2,
    folds=["fold_1", "fold_2"],
    custom_label_choice="semantic",
    sampler=MinTwoInstanceSampler(),
    label_dtype=label_dtype,
    raw_transform=raw_transform,
    download=True,
    transform=transform
)


def visualize_transformations():
    image_path = os.path.join(result_dir, "data_visualisation_transforms", "images")
    label_path = os.path.join(result_dir, "data_visualisation_transforms", "labels")
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    for idx, (image, label) in enumerate(semantic_dataset, start=1):
        image = image.numpy()
        label = label.numpy()
        print(len(np.unique(image)))
        image = image.transpose(1, 2, 0)
        label = np.squeeze(label)

        imageio.imwrite(os.path.join(image_path, f"{idx:04}.tiff"), image)
        imageio.imwrite(os.path.join(label_path, f"{idx:04}.tiff"), label)
        if idx == 30:
            break

# import h5py
# h5_paths = get_pannuke_paths(data_path)
# for h5_path in h5_paths:
#     with h5py.File(h5_path, "r") as file:
#         instances = file["labels/instances"][:]
#         semantics = file["labels/semantic"][:]
#         for instance_label, semantic_label in zip(instances, semantics):

visualize_transformations()
result_dict = {
    "gamma": [],
    "average unique indices drawn": [],
    "covered samples in 10 iterations": [],
}


def check_sampled_indices():
    for gamma in np.linspace(0.5, 1, 6):
        sampler = get_sampler(instance_dataset, semantic_dataset, gamma=1, path=data_path, split="train")

        uniques_per_sampler = []
        uniques_all_samplers = []
        for i in range(10):
            indices = []
            for idx in sampler:
                indices.append(idx)
            uniques_per_sampler.append(len(np.unique(indices)))
            uniques_all_samplers.extend(np.unique(indices).tolist())
        print(f"Unique indices for gamma {gamma}: {uniques_per_sampler}, {len(sampler)}")
        print(f"Over 10 samplers unique indices: {len(np.unique(uniques_all_samplers))}")
        result_dict["gamma"].append(gamma)
        result_dict["average unique indices drawn"].append(np.mean(uniques_per_sampler))
        result_dict["covered samples in 10 iterations"].append(len(np.unique(uniques_all_samplers)))

    df = pd.DataFrame(result_dict)
    df.to_csv(os.path.join(result_dir, "gamma_weighted_sampling.csv"), index=False)


def check_sampled_instances():
    results_dict = {
        "gamma": [],
        "1": [],
        "2": [],
        "3": [],
        "4": [],
        "5": [],
    }
    # for gamma in np.linspace(0.5, 1, 6):
    #     sampler = get_sampler(random_inst_dataset, random_sem_dataset, gamma=gamma, path=data_path, split="train")
    #     result_array = np.array([
    #         [len(np.unique(random_inst_dataset[index][1][random_sem_dataset[index][1] == cell_type])) for cell_type in range(1, 6)] for index in tqdm(sampler)
    #     ])
    #     # for index in sampler:
    #     #     _, semantic_label = semantic_dataset[index]
    #     #     _, instance_label = instance_dataset[index]
    #     #     result_array.append([len(np.unique(instance_label[semantic_label == cell_type]))
    #     #           for cell_type in range(1, 6)])
    #     # result_array = np.array(result_array)
    #     class_instance_counts = np.sum(result_array, axis=0).tolist()
    #     print(f"Weighted sampling for gamma {gamma}: {class_instance_counts}")
    #     results_dict["gamma"].append(gamma)
    #     results_dict["1"].append(class_instance_counts[0])
    #     results_dict["2"].append(class_instance_counts[1])
    #     results_dict["3"].append(class_instance_counts[2])
    #     results_dict["4"].append(class_instance_counts[3])
    #     results_dict["5"].append(class_instance_counts[4])



    sampler = RandomSampler(semantic_dataset)
    for i in range(10):
        result_array = np.array([
            [len(np.unique(random_inst_dataset[index][1][random_sem_dataset[index][1] == cell_type])) for cell_type in range(1, 6)] for index in tqdm(sampler)
            ])
        class_instance_counts = np.sum(result_array, axis=0).tolist()
        print(f"Random sampling: {class_instance_counts}")
        results_dict["gamma"].append(str(i))
        results_dict["1"].append(class_instance_counts[0])
        results_dict["2"].append(class_instance_counts[1])
        results_dict["3"].append(class_instance_counts[2])
        results_dict["4"].append(class_instance_counts[3])
        results_dict["5"].append(class_instance_counts[4])

    df = pd.DataFrame(results_dict)
    print(df.head())
    df.to_csv(os.path.join(result_dir, "per_class_instances_gamma_random_mininstance.csv"), index=False)

# check_sampled_instances()
# check_sampled_indices()


# check_sampled_instances()


# TODO emply logic to check how many indices are actually succcessfully sampled with a RandomSampler, 
# i. e. surpassing the MinTwoInstanceSampler! --> maybe try hashing the images somehow and work with a set to
# check for unique samples


def get_array_hash(array) -> str:
    data = bytes()
    data += array.numpy().tobytes()

    return hashlib.shake_256(data).hexdigest(16).upper()


def check_random_indices():
    sampler = RandomSampler(random_sem_dataset)
    unique_indices = [len(set([get_array_hash(random_sem_dataset[idx][1]) for idx in sampler])) for i in tqdm(range(10))]
    print(unique_indices)
    idx_dict = {"unique_samples": unique_indices}
    df = pd.DataFrame(idx_dict)
    df.to_csv(os.path.join(result_dir, "random_sampled_unique_indices.csv"), index=False)

# check_random_indices()