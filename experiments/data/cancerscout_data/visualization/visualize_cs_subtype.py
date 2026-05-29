import argparse
from pathlib import Path

import h5py
import napari
import pandas as pd

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data")
metadata_csv = ROOT / "cancerscout_metadata" / "cancerscout_semantic_organized.csv"


def visualize_cancerscout(input_path: Path, split, subtype, healthy):
    tissue_type = "new_tumor" if not healthy else "new_non_tumor"
    all_h5_files = list(input_path.glob(f"{split}_models/{tissue_type}_data/fixed_h5_files/*.h5"))
    df = pd.read_csv(metadata_csv, index_col="filename")
    for h5_path in all_h5_files:
        if subtype:
            if df.loc[h5_path.stem, "subtype"] != subtype:
                continue
        with h5py.File(h5_path, "r") as f:
            img = f["img"][:].transpose(1, 2, 0)
            label = f["inst_labels/v_2_pproc"][:]
            ignite_pred = f["ignite_pred"][:]
        viewer = napari.Viewer()
        viewer.add_image(img, name="img")
        viewer.add_labels(label, name="inst_label")
        viewer.add_labels(ignite_pred, name="ignite_pred")
        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", default=ROOT)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--healthy", action="store_true")
    parser.add_argument("--subtype", type=str, default=None)
    args = parser.parse_args()
    visualize_cancerscout(Path(args.input_path), args.split, args.subtype, args.healthy)
