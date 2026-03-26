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
from micro_sam.object_classification import compute_object_features

H5_DIR = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/new_tumor_data"
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


def start_object_feature_computation(seg):
    print("Starting computation of object features in the background.")
    state = AnnotatorState()
    start = time.time()
    seg_ids, features = compute_object_features(state.image_embeddings, seg)
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

    needed_keys = ["img", "inst_label"]

    for input_path, embedding_path in zip(input_paths, embedding_paths):
        img_name = os.path.basename(embedding_path)
        state = AnnotatorState()
        with h5py.File(input_path, "r") as f:

            if not all(key in f.keys() for key in needed_keys):
                print(f"Missing keys: {[key for key in needed_keys if key not in f.keys()]}")
                continue
            img = f["img"][:]
            seg = f["inst_label"][:]

            if "object_features" and "seg_ids" in f.keys():
                state.seg_ids = f["object_features"][:]
                state.object_features = f["seg_ids"][:]
            else:
                threading.Thread(target=start_object_feature_computation, args=(seg,))

            ign_pred = f["ignite_pred"] if "ignite_pred" in f.keys() else None

        viewer = get_object_classifier_viewer(
            image=img,
            segmentation=seg,
            embedding_path=embedding_path,
        )

        if ign_pred is not None:
            viewer.add_labels(ign_pred, name="Ignite prediction", opacity=0.6)

        viewer.add_shapes(get_grid(seg.shape), shape_type="line", name=img_name, edge_width=3)
        state = AnnotatorState()

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

        state.features_output = os.path.join(rf_dir, f"{img_name.split('-')[0]}_features.npy")
        state.labels_output = os.path.join(rf_dir, f"{img_name.split('-')[0]}_labels.npy")

        os.makedirs(rf_dir, exist_ok=True)
        state.rf_dir = rf_dir

        # TODO: Add versioning for rf models and cached training data, so they can be
        # selected for training and model loading
        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, default=H5_DIR)
    parser.add_argument("--embedding_dir", "-e", type=str, default=EMB_DIR)
    parser.add_argument("--rf_dir", type=str, default=RF_DIR)

    args = parser.parse_args()
    semantic_annotation(
        input_dir=args.input_dir,
        embedding_dir=args.embedding_dir,
        rf_dir=args.rf_dir,
    )


if __name__ == "__main__":
    main()
