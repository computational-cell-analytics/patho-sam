import micro_sam.training as sam_training
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import (
    get_consep_loader,
    get_cpm_loader,
    get_cryonuseg_loader,
    get_glas_loader,
    get_lizard_loader,
    get_lynsec_loader,
    get_monusac_loader,
    get_monuseg_loader,
    get_nuclick_loader,
    get_nuinsseg_loader,
    get_pannuke_loader,
    get_puma_loader,
    get_srsanet_loader,
    get_tnbc_loader,
)
from raw_loader import get_loader

def custom_transform(x, y):
    return x, y


def get_dataloaders(patch_shape, data_path, dataset, split=None, organ_type=None):
    raw_transform = sam_training.identity
    sampler = MinInstanceSampler(min_num_instances=3)

    if dataset == "consep":
        loader = get_consep_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            split=split,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "cpm15":
        loader = get_cpm_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=False,
            split=split,
            raw_transform=raw_transform,
            sampler=sampler,
            data_choice="cpm15",
        )

    elif dataset == "cpm17":
        loader = get_cpm_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=False,
            split=split,
            raw_transform=raw_transform,
            sampler=sampler,
            data_choice="cpm17",
        )

    elif dataset == "cryonuseg":
        loader = get_cryonuseg_loader(
            path=data_path,
            patch_shape=(1,) + patch_shape,
            batch_size=1,
            rater="b1",
            split=split,
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "glas":
        loader = get_glas_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            split=split,
            raw_transform=raw_transform,
            sampler=MinInstanceSampler(min_num_instances=2),
        )

    elif dataset == "lizard":
        loader = get_lizard_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            split=split,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "lynsec_he":
        loader = get_lynsec_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            choice="h&e",
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "lynsec_ihc":
        loader = get_lynsec_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            choice="ihc",
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "monusac":
        loader = get_monusac_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            split=split,
            download=True,
            organ_type=organ_type,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "monuseg":
        loader = get_monuseg_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            split=split,
            download=True,
            organ_type=organ_type,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "nuinsseg":
        loader = get_nuinsseg_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
        )
    elif dataset == "nuclick":
        loader = get_nuclick_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            split=split,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "pannuke":
        loader = get_pannuke_loader(
            path=data_path,
            patch_shape=(1,) + patch_shape,
            batch_size=1,
            folds=split,
            shuffle=False,
            ndim=2,
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
        )
    elif dataset == "pannuke_sem":
        loader = get_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            sampler=MinInstanceSampler(min_num_instances=3),
            transform=custom_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "puma":
        loader = get_puma_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            annotations="nuclei",
            download=True,
            split=split,
            raw_transform=raw_transform,
            sampler=sampler,
        )


    elif dataset == "srsanet":
        loader = get_srsanet_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            split=split,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "tnbc":
        loader = get_tnbc_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            ndim=2,
            download=True,
            split=split,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    return loader
