import os
import shutil
from glob import glob
from tqdm import tqdm

import numpy as np
import imageio.v3 as imageio


DATASETS = [
    "consep",
    "cpm15",
    "cpm17",
    "cryonuseg",
    "lizard",
    "lynsec_he",
    "lynsec_ihc",
    "monusac",
    "monuseg",
    "nuclick",
    "nuinsseg",
    "pannuke",
    "puma",
    "srsanet",
    "tnbc",
]


def preprocess_cvtplus(input_dir, output_dir):
    # create resized pyramid tiffs with vips
    import pyvips

    for dataset in DATASETS:
        data_dir = os.path.join(input_dir, dataset, "loaded_testset", "eval_split", "test_images")
        intermediate_folder = os.path.join(output_dir, "intermediate", dataset)
        output_folder = os.path.join(output_dir, "preprocessed", dataset)
        os.makedirs(intermediate_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        for img_path in tqdm(glob(os.path.join(data_dir, "*.tiff"))):
            img = imageio.imread(img_path)
            img_uint = img.astype(np.uint8)
            intermediate_file = os.path.join(intermediate_folder, os.path.basename(img_path))
            imageio.imwrite(intermediate_file, img_uint)
            output_file = os.path.join(output_folder, os.path.basename(img_path))
            image = pyvips.Image.new_from_file(intermediate_file)
            image.tiffsave(output_file, tile=True, tile_width=512, tile_height=512, pyramid=True)

    shutil.rmtree(intermediate_folder)


def main():
    preprocess_cvtplus(
        input_dir="/mnt/lustre-grete/usr/u12649/data/final_test", output_dir="/mnt/lustre-grete/usr/u12649/data/cvtplus"
    )


if __name__ == "__main__":
    main()
