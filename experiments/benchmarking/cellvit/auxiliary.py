import os
import shutil


def remove_cellvit_plots(path):
    for dataset in [
        "cpm15",
        "cpm17",
        "cryonuseg",
        "janowczyk",
        "lizard",
        "lynsec",
        "monusac",
        "monuseg",
        "nuinsseg",
        "pannuke",
        "puma",
        "tnbc",
    ]:
        for model in ["256-x20", "256-x40", "SAM-H-x20", "SAM-H-x40"]:
            plot_dir = os.path.join(path, dataset, model, dataset, "plots")
            if os.path.exists(plot_dir):
                shutil.rmtree(plot_dir)


remove_cellvit_plots("/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/inference")
