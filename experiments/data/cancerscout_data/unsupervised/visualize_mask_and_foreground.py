from pathlib import Path

import imageio.v3 as imageio
import napari
import pandas as pd
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
DARK_AREA_SAMPLES = ["A2020-001401_1-1-1_HE-2021-10-11T16-57-55"]

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/univ2/output_mpp_2")
CSV_PATH = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/cancerscout_metadata/cancerscout_organized.csv"


def visualize_extracted_wsis():
    img_dirs = list(ROOT.glob("*"))
    df = pd.read_csv(CSV_PATH, index_col="filename")

    for img_dir in tqdm(img_dirs):
        if img_dir.name not in DARK_AREA_SAMPLES:
            continue
        mask_path = img_dir / "tissue_mask.png"
        img_path = img_dir / "thumbnail.png"
        assert img_path.exists() and mask_path.exists()
        viewer = napari.Viewer()
        viewer.add_image(imageio.imread(img_path), name=img_dir.name)
        viewer.add_labels(imageio.imread(mask_path), name=df.loc[img_dir.name, "subtype"])
        napari.run()


visualize_extracted_wsis()
