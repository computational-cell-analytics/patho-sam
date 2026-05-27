import argparse
import json
from collections import defaultdict
from multiprocessing import cpu_count
from pathlib import Path

import h5py
import napari
from magicgui import magicgui
from micro_sam.instance_segmentation import get_predictor_and_decoder
from micro_sam.sam_annotator.object_classifier import AnnotatorState, project_prediction_to_segmentation
from micro_sam.util import precompute_image_embeddings
from natsort import natsorted
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from util import ClassCountWidget, get_grid, get_object_classifier_viewer

from patho_sam.annotation import compute_object_features_parallel, postprocess_instance_mask

labels = {
    "Unannotated": 0,
    "Background": 1,
    "Tumor cells": 2,
    "Reactive epithelium": 3,
    "Stroma": 4,
    "Inflammation": 5,
    "Alveolar tissue": 6,
    "Fatty tissue": 7,
    "Necrotic tissue": 8,
    "Erythrocytes": 9,
    "Bronchial epithelium": 10,
    "Mucus/Plasma/Fluids": 11,
    "Cartilage/Bone": 12,
    "Macrophages": 13,
    "Muscle": 14,
    "Liver": 15,
    "Keratinization": 16,
}


H5_DIR = "/mnt/ceph-hdd/cold/nim00020/hannibal_data"
LABELS = {"Tumor cells": 1, "Stroma cells": 2, "Lymphocytes": 3, "Others": 4, "Neutrophils": 5, "Epithelial": 6}


def semantic_annotation(h5_dir: Path, embedding_dir: Path, correction: int = 0):
    required_keys = ["img", "inst_labels"]

    json_path = h5_dir.parent / f"{h5_dir.parent.parent.name.split('_')[0]}_{h5_dir.parent.name}.json"

    if json_path.exists() and json_path.stat().st_size > 0:
        with open(json_path, "r") as f:
            raw = json.load(f)
        file_dict = {k: set(v) for k, v in raw.items()}
        file_dict = defaultdict(set, file_dict)

    else:
        file_dict = defaultdict(set)

    predictor, _ = get_predictor_and_decoder(device="cpu", model_type="vit_b_histopathology")
    state = AnnotatorState()
    state.json_path = json_path
    state.file_dict = file_dict

    for input_path in h5_dir.glob("*.h5"):
        embedding_path = embedding_dir / input_path.stem
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embeddings not stored at {embedding_path}")

        state.h5_path = input_path

        if input_path.name in state.file_dict:
            if correction in state.file_dict[input_path.name]:
                print(50 * "-", f"\n Training data v{correction} already exists for {input_path.name}")
                continue

        with h5py.File(input_path, "r") as f:
            if not all(key in f.keys() for key in required_keys):
                print(f"Missing keys: {[key for key in required_keys if key not in f.keys()]}")
                continue

            img = f["img"][:].transpose(1, 2, 0)
            embeddings = precompute_image_embeddings(
                predictor, img, embedding_path, ndim=2, tile_shape=(384, 384), halo=(64, 64)
            )

            latest_seg = natsorted([key for key in f["inst_labels"].keys() if "prelim" not in key])
            seg_to_load = "v_2_pproc" if "v_2_pproc" in latest_seg else latest_seg[-1]
            seg = f[f"inst_labels/{seg_to_load}"][:]
            print(f"Loaded {seg_to_load} as segmentation")
            if seg_to_load != "v_2_pproc":
                print("Warning: unexpected segmentation loaded!")

            if "all_features" in f:
                state.seg_ids = f["all_seg_ids"][:]
                state.object_features = f["all_features"][:]
            else:
                seg = postprocess_instance_mask(seg, area_threshold=50)
                state.seg_ids, state.object_features = compute_object_features_parallel(
                    segmentation=seg, image_embeddings=embeddings, n_workers=max(1, cpu_count())
                )

            if "train_labels" in f:
                train_labels = f["train_labels"][:]
                train_seg_ids = f["train_seg_ids"][:]

            else:
                train_labels, train_seg_ids = None, None

            ign_pred = f["ignite_pred"][:] if "ignite_pred" in f.keys() else None

        viewer = get_object_classifier_viewer(
            image=img,
            segmentation=seg,
            embedding_path=embedding_path,
            interpolation2d="linear",
            contrast_limits=(0, 255),
            rendering="mip",
        )

        if train_labels is not None:
            annotation_data = project_prediction_to_segmentation(seg, train_labels, train_seg_ids)
            viewer.layers["annotations"].data = annotation_data
            print("Loaded previous annotations!")

        if ign_pred is not None:
            viewer.add_labels(ign_pred, name="Ignite prediction", opacity=0.6)

        viewer.add_shapes(get_grid(seg.shape), shape_type="line", name=input_path.stem, edge_width=3)

        @magicgui(call_button="Save definite training data to version")
        def save_annotation(viewer: "napari.Viewer"):
            state = AnnotatorState()
            state.file_dict[input_path.name].add(correction)
            _file_dict = {key: list(value) for key, value in state.file_dict.items()}
            with open(state.json_path, "w") as f:
                json.dump(_file_dict, f, indent=4)
            print(f"Saved correction {correction} for {input_path.name}")

        widget = QWidget()
        layout = QVBoxLayout()

        for name, value in LABELS.items():
            button = QPushButton(name)
            labels = viewer.layers["annotations"]

            def make_callback(v):
                return lambda: setattr(labels, "selected_label", v)

            button.clicked.connect(make_callback(value))

            layout.addWidget(button)

        widget.setLayout(layout)
        count_widget = ClassCountWidget(viewer=viewer, labels=LABELS, annotation_layer_name="annotations")
        viewer.window.add_dock_widget(save_annotation, area="right")
        viewer.window.add_dock_widget(count_widget, area="right", name="Class Counts")
        viewer.window.add_dock_widget(widget, area="right")
        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, default=H5_DIR)
    parser.add_argument("--healthy", action="store_true")
    parser.add_argument("--split", "-s", type=str, default="train")
    parser.add_argument("--correction", "-c", type=int, default=0)

    args = parser.parse_args()
    entity = "tumor" if not args.healthy else "non_tumor"
    embeddings_dir = Path(args.input_dir) / f"{args.split}_models" / f"new_{entity}_data" / "embeddings"
    h5_dir = Path(args.input_dir) / f"{args.split}_models" / f"new_{entity}_data" / "fixed_h5_files"
    semantic_annotation(h5_dir=h5_dir, embedding_dir=embeddings_dir, correction=args.correction)


if __name__ == "__main__":
    main()
