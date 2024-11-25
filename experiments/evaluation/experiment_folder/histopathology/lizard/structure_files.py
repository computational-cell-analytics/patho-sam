import os
import shutil
from glob import glob
import argparse
from natsort import natsorted
import torch
from skimage import io
import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model


def label_check(path):
    for label in natsorted(os.listdir(path)):
        print(label.size())

label_check('/mnt/lustre-grete/usr/u12649/scratch/data/lizard/loaded_dataset/complete_dataset/labels')


