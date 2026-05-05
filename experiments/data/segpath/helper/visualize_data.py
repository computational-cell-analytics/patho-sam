import napari
from pathlib import Path
import pandas as pd 
import h5py


ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/segpath")

CELL_NAMES = {
    # "smooth_muscle": "aSMA_SmoothMuscle",
              "endothelium": "ERG_Endothelium",
            #   "lymphocytes": "CD3CD20_Lymphocyte",
            #   "epithelium": "panCK_Epithelium"
              }

for cell_type in CELL_NAMES.values():
    data_dir = ROOT / cell_type / "data"
    csv_path = ROOT / cell_type / f"{cell_type}_summary.csv"
    df = pd.read_csv(csv_path)
    df = df[df["training_objects"] > 0]
    df = df.set_index("filename")
    predicted_samples = df.index.tolist()

    paths = [data_dir / pred_sample for pred_sample in predicted_samples]

    i = 0

    for path in paths:
        with h5py.File(path, 'r') as f:
            pred = f["labels/postprocessed_pred"][:]
            bin_mask = f["labels/best_crop"][:]
            img = f["images/best_crop"][:]
            features = f["object_features"][:]

        viewer = napari.Viewer()
        viewer.add_image(img, name=path.stem)
        viewer.add_labels(pred, name="segpath_prediction")
        viewer.add_labels(bin_mask, name="binary label")
        napari.run()
