from torch_em.data.datasets.histopathology.cpm import get_cpm_loader
from torch_em.data.datasets.histopathology.cryonuseg import get_cryonuseg_loader
from torch_em.data.datasets.histopathology.janowczyk import get_janowczyk_loader 
# from torch_em.data.datasets.histopathology.lizard import get_lizard_loader
from torch_em.data.datasets.histopathology.lynsec import get_lynsec_loader
from torch_em.data.datasets.histopathology.monusac import get_monusac_loader
from torch_em.data.datasets.histopathology.monuseg import get_monuseg_loader
from torch_em.data.datasets.histopathology.nuinsseg import get_nuinsseg_loader
from torch_em.data.datasets.histopathology.pannuke import get_pannuke_loader
# from torch_em.data.datasets.histopathology.puma import get_puma_loader 
from puma1 import get_puma_loader
from lizard import get_lizard_loader
# from torch_em.data.datasets.histopathology.tnbc import get_tnbc_loader
from tnbc import get_tnbc_loader
import os
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training
""" This loads the selected datasets as .tiff files with an image shape of (512, 512, 3) and a label shape of (512, 512)"""


def get_dataloaders(patch_shape, data_path, dataset, split=None, organ_type=None):
    print('Dataloaders.py is running')
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
    sampler = MinInstanceSampler(min_num_instances=3)
    if dataset == 'cpm15':
        loader = get_cpm_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=False,
            raw_transform=raw_transform,
            sampler=sampler,
            data_choice='cpm15'
        )
    elif dataset == 'cpm17':
        loader = get_cpm_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=False,
            raw_transform=raw_transform,
            sampler=sampler,
            data_choice='cpm17'
        )
    elif dataset == 'cryonuseg': ###cave has to iterate over rater!
        loader = get_cryonuseg_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            rater=split,
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
            )
    elif dataset == 'janowczyk': 
        loader = get_janowczyk_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            raw_transform=raw_transform,
            annotation='nuclei',
            sampler=sampler,
            )
    elif dataset == 'lizard': ###cave has to iterate over split!
        loader = get_lizard_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            split=split,
            raw_transform=raw_transform,
            sampler=sampler,
            )
    elif dataset == 'lynsec': 
        loader = get_lynsec_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            choice='h&e',
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
            )
    elif dataset == 'monusac': ###cave has to iterate over train / test!
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
    elif dataset == 'monuseg': ###cave has to iterate over train / test!
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
    elif dataset == 'nuinsseg': 
        loader = get_nuinsseg_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
            )
    elif dataset == 'pannuke': ###cave enables iteration, depending on what folds are supposed to be loaded; for test --> fold3
        loader = get_pannuke_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            folds=split,
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
            )
    elif dataset == 'puma': 
        loader = get_puma_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            annotations='nuclei',
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
            )
    elif dataset == 'tnbc': 
        loader = get_tnbc_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            download=True,
            raw_transform=raw_transform,
            sampler=sampler,
            )
    return loader

if __name__ == "__main__":
    get_dataloaders()