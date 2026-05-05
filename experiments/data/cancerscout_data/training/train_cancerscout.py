import argparse

import torch
import torch.utils.data as data_util
import torch_em
from _cancerscout_dataset import get_cancerscout_dataset
from micro_sam.training import train_sam
from torch_em.data import MinInstanceSampler
from torch_em.transform.label import PerObjectDistanceTransform

from patho_sam.training import histopathology_identity

ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/"
SAVE_ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/pathosam-models/cancerscout_instance"


def _get_train_val_split(ds, val_fraction=0.2):
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = data_util.random_split(ds, [1 - val_fraction, val_fraction], generator=generator)
    return train_ds, val_ds


def get_dataloaders(path, patch_shape) -> torch.utils.data.DataLoader:

    label_dtype = torch.float32
    sampler = MinInstanceSampler(min_num_instances=10)

    # Expected raw and label transforms.
    raw_transform = histopathology_identity
    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=10
    )

    cancerscout_dataset = get_cancerscout_dataset(
        path=path,
        entities=["tumor", "non_tumor"],
        patch_shape=patch_shape,
        split="train",
        label_dtype=label_dtype,
        raw_transform=raw_transform,
        label_transform=label_transform,
        sampler=sampler,
    )

    train_ds, val_ds = _get_train_val_split(cancerscout_dataset, val_fraction=0.2)

    train_loader = torch_em.get_data_loader(train_ds, batch_size=1, shuffle=True, num_workers=16)

    val_loader = torch_em.get_data_loader(val_ds, batch_size=1, shuffle=False, num_workers=16)

    return train_loader, val_loader


def train_cancerscout(data_path, save_root, iterations, model_type):

    train_loader, val_loader = get_dataloaders(data_path, patch_shape=(512, 512))

    train_sam(
        name="pathosam-cancerscout-instance",
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_iterations=iterations,
        save_root=save_root,
        early_stopping=None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str, default=ROOT)
    parser.add_argument("--model_type", default="vit_b_histopathology", help="Model to finetune")
    parser.add_argument("--save_root", "-s", type=str, default=SAVE_ROOT)
    parser.add_argument("--n_iterations", type=int, default=1e5)
    args = parser.parse_args()
    train_cancerscout(
        data_path=args.input_path, save_root=args.save_root, iterations=args.n_iterations, model_type=args.model_type
    )


if __name__ == "__main__":
    main()
