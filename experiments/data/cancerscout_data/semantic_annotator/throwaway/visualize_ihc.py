from pathlib import Path
import h5py
import napari
import imageio.v3 as imageio
ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/rois_pdl1_ihc/h5_files"

for h5_path in Path(ROOT).glob("*.h5"):
    viewer = napari.Viewer()
    with h5py.File(h5_path, 'r') as f:
        img = f["img"][:]
    viewer.add_image(img, name=h5_path.stem.split("_")[0])
    label_path = h5_path.parent.parent / "segmentations_v3" / (str(h5_path.stem) + ".tif")
    pred = imageio.imread(label_path)
    viewer.add_labels(pred, name="PathoSAM pred")
    napari.run()
