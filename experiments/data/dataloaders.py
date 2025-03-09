import micro_sam.training as sam_training

from torch_em.data import datasets
from torch_em.data import MinInstanceSampler


def get_dataloaders(patch_shape, data_path, dataset):
    raw_transform = sam_training.identity
    sampler = MinInstanceSampler(min_num_instances=3)

    if dataset == "consep":
        loader = datasets.get_consep_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            split="test",
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "cpm15":
        loader = datasets.get_cpm_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=False,
            split="test",
            raw_transform=raw_transform,
            sampler=sampler,
            data_choice="cpm15",
        )

    elif dataset == "cpm17":
        loader = datasets.get_cpm_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=False,
            split="test",
            raw_transform=raw_transform,
            sampler=sampler,
            data_choice="cpm17",
        )

    elif dataset == "cryonuseg":
        loader = datasets.get_cryonuseg_loader(
            path=data_path,
            patch_shape=(1,) + patch_shape,
            batch_size=1,
            rater="b1",
            split="test",
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "glas":
        loader = datasets.get_glas_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            split="test",
            raw_transform=raw_transform,
            sampler=MinInstanceSampler(min_num_instances=2),
        )

    elif dataset == "lizard":
        loader = datasets.get_lizard_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            split="test",
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "lynsec_he":
        loader = datasets.get_lynsec_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            choice="h&e",
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "lynsec_ihc":
        loader = datasets.get_lynsec_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            choice="ihc",
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "monusac":
        loader = datasets.get_monusac_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            split="test",
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "monuseg":
        loader = datasets.get_monuseg_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            split="test",
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "nuinsseg":
        loader = datasets.get_nuinsseg_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
        )
    elif dataset == "nuclick":
        loader = datasets.get_nuclick_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            split="Validation",
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "pannuke":
        loader = datasets.get_pannuke_loader(
            path=data_path,
            patch_shape=(1,) + patch_shape,
            batch_size=1,
            folds=["fold_3"],
            ndim=2,
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "puma":
        loader = datasets.get_puma_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            annotations="nuclei",
            download=True,
            split="test",
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "srsanet":
        loader = datasets.get_srsanet_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            split="test",
            raw_transform=raw_transform,
            sampler=sampler,
        )

    elif dataset == "tnbc":
        loader = datasets.get_tnbc_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            ndim=2,
            download=True,
            split="test",
            raw_transform=raw_transform,
            sampler=sampler,
        )

    return loader
