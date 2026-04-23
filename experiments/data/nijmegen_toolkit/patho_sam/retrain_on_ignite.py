import os 
import argparse
from natsort import natsorted
from glob import glob
import torch_em
from torch_em.data import MinInstanceSampler
from micro_sam.training import get_trainable_sam_model
from patho_sam.training import histopathology_identity

INPUT_ROOT = ""


def get_image_and_label_paths(data_dir):
    image_paths = []
    label_paths = []
    return image_paths, label_paths


def get_dataloaders(data_dir, batch_size):
    image_paths, label_paths = get_image_and_label_paths(data_dir)
    sampler = MinInstanceSampler(min_num_instances=10)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        label_paths=label_paths,
        sampler=sampler,
        raw_transform=histopathology_identity,
        ndim=2,
        with_channels=True,
    )
    train_loader = torch_em.get_data_loader(batch_size=batch_size, shuffle=True)
    val_loader = torch_em.get_data_loader(batch_size=1, shuffle=True)

    return train_loader, val_loader


def finetune_ihc_model(input_dir, batch_size, save_root):
    train_loader, val_loader = get_dataloaders(input_dir, batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=int, default=INPUT_ROOT)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--save_root", "-s", type=str)

    args = parser.parse_args()
    finetune_ihc_model(input_dir=args.input_dir,
                       batch_size=args.patch_size,
                       save_root=args.save_root)


if __name__ == "__main__":
    main()
