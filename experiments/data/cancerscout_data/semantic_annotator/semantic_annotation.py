import os
import argparse
from natsort import natsorted
from glob import glob
import h5py
import napari
import time
from magicgui import magicgui
import numpy as np
import threading
from micro_sam.sam_annotator.object_classifier import object_classifier, AnnotatorState
from micro_sam import util
from patho_sam.annotation import compute_object_features_parallel

H5_DIR = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/new_tumor_data/new_h5_files"
EMB_DIR = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/new_tumor_data/embeddings"
RF_DIR = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/new_tumor_data/rf_output"


def get_grid(img_shape, cell_width=512):
    """Add Grid for better overview in annotated regions"""
    return [
        np.array([
            [0, cell_width * i],
            [img_shape[0], cell_width * i]
        ])
        for i in range(0, img_shape[1] // cell_width + 1)
        ] + [
        np.array([
            [cell_width * i, 0],
            [cell_width * i, img_shape[1]]
        ])
        for i in range(0, img_shape[0] // cell_width + 1)
    ]


def get_object_classifier_viewer(image, segmentation, embedding_path):
    tile_shape, halo = (384, 384), (64, 64)
    viewer = object_classifier(
        image=image,
        segmentation=segmentation,
        model_type="vit_b_histopathology",
        embedding_path=embedding_path,
        tile_shape=tile_shape,
        halo=halo,
        ndim=2,
        return_viewer=True
    )
    return viewer


def start_object_feature_computation(seg, embeddings):
    print("Starting computation of object features in the background.", flush=True)
    state = AnnotatorState()
    start = time.time()
    seg_ids, features = compute_object_features_parallel(seg, embeddings, n_workers=os.cpu_count()-3)
    state.seg_ids = seg_ids
    state.object_features = features
    end = time.time()
    print(f"Computation of object features for {len(seg_ids)} took {end - start:.4f} seconds")


def semantic_annotation(input_dir, embedding_dir, rf_dir):
    input_paths = natsorted(glob(os.path.join(input_dir, "*.h5")))
    embedding_paths = natsorted(glob(os.path.join(embedding_dir, "*")))

    os.makedirs(rf_dir, exist_ok=True)

    # Check if embeddings, segmentations and
    if not len(input_paths) == len(embedding_paths):
        raise ValueError(
            f"Inconsistent input: {len(input_paths)} imgs, {len(embedding_paths)} embs")

    needed_keys = ["img", "inst_labels"]

    predictor, _ = util.get_sam_model(
                    device="cpu", model_type="vit_b_histopathology",
                    checkpoint_path=None, decoder_path=None, return_state=True,
                    progress_bar_factory=None,
                )

    for input_path, embedding_path in zip(input_paths, embedding_paths):
        img_name = "_".join(os.path.basename(embedding_path).split("_")[:2])
        state = AnnotatorState()
        state.img_name = img_name
        with h5py.File(input_path, "r") as f:

            if not all(key in f.keys() for key in needed_keys):
                print(f"Missing keys: {[key for key in needed_keys if key not in f.keys()]}")
                continue
            img = f["img"][:]
            embeddings = util.precompute_image_embeddings(predictor, img, embedding_path, ndim=2,
                                                          tile_shape=(384, 384), halo=(64, 64))

            latest_seg = natsorted([key for key in f["inst_labels"].keys() if "prelim" not in key])[-1]
            seg = f[f"inst_labels/{latest_seg}"][:]
            t = threading.Thread(target=start_object_feature_computation, args=(seg, embeddings,))
            t.start()
            ign_pred = f["ignite_pred"][:] if "ignite_pred" in f.keys() else None

        viewer = get_object_classifier_viewer(
            image=img,
            segmentation=seg,
            embedding_path=embedding_path,
        )

        if ign_pred is not None:
            viewer.add_labels(ign_pred, name="Ignite prediction", opacity=0.6)

        viewer.add_shapes(get_grid(seg.shape), shape_type="line", name=img_name, edge_width=3)

        @magicgui(call_button="Save definite annotation")
        def save_annotation(viewer: "napari.Viewer"):
            layer_name = "prediction"

            if layer_name not in viewer.layers:
                print("layer not found")
                return

            definite_ann = viewer.layers[layer_name].data
            with h5py.File(input_path, "a") as f:
                f.create_dataset("sem_label", definite_ann)
        viewer.window.add_dock_widget(save_annotation, area="right")

        state.train_data_path = os.path.join(rf_dir, "train_data.h5")

        os.makedirs(rf_dir, exist_ok=True)
        state.rf_dir = rf_dir
        state.train_data_path = os.path.join(rf_dir, "rf_training_data.h5")
        napari.run()


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_dir", "-i", type=str, default=H5_DIR)
    parser.add_argument("--healthy", action="store_true")
    # parser.add_argument("--embedding_dir", "-e", type=str, default=EMB_DIR)
    parser.add_argument("--rf_dir", type=str, default=RF_DIR)
    args = parser.parse_args()
    entity = "tumor" if not args.healthy else "non_tumor"
    H5_DIR = f"/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/new_{entity}_data/new_h5_files"
    EMB_DIR = f"/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/new_{entity}_data/embeddings"
    semantic_annotation(
        input_dir=H5_DIR,
        embedding_dir=EMB_DIR,
        rf_dir=args.rf_dir,
    )


if __name__ == "__main__":
    main()
