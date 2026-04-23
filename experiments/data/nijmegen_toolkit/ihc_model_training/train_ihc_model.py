import os
import torch
import argparse

from micro_sam.training import train_instance_segmentation
from get_nijmegen_loaders import get_ignite_loader
from torch_em.data import MinInstanceSampler
from torch_em.transform.label import PerObjectDistanceTransform
from patho_sam.training import histopathology_identity


def get_dataloaders(path, patch_shape) -> torch.utils.data.DataLoader:

    label_dtype = torch.float(32)
    sampler = MinInstanceSampler(min_num_instances=10)

    # Expected raw and label transforms.
    raw_transform = histopathology_identity
    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=10,
    )

    train_loader = get_ignite_loader(path,
                                     batch_size=2,
                                     patch_shape=patch_shape,
                                     split="train",
                                     sampler=sampler,
                                     label_dtype=label_dtype,
                                     raw_transform=raw_transform,
                                     label_transform=label_transform,
                                     shuffle=True,
                                     num_workers=16
                                     )

    val_loader = get_ignite_loader(path,
                                   batch_size=1,
                                   patch_shape=patch_shape,
                                   split="val",
                                   sampler=sampler,
                                   label_dtype=label_dtype,
                                   raw_transform=raw_transform,
                                   label_transform=label_transform,
                                   shuffle=False,
                                   num_workers=16
                                   )

    return train_loader, val_loader


def train_ihc_ignite(data_path, save_root, iterations, model_type):
    train_loader, val_loader = get_dataloaders(data_path, patch_shape=(512, 512))

    train_instance_segmentation(
        name="pathosam-ignite-instance",
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_iterations=iterations,
        lr=5e-5,
        save_root=save_root,
        early_stopping=None
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str)
    parser.add_argument("--model_type", default="vit_b_histopathology", help="Model to finetune")
    parser.add_argument("--save_root", "-s", type=str)
    parser.add_argument("--n_iterations", type=int, default=5e4)
    args = parser.parse_args()
    train_ihc_ignite(
        data_path=args.input_path,
        save_root=args.save_root,
        iterations=args.n_iterations,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
