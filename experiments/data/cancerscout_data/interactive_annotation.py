import argparse
from pathlib import Path

import h5py
import napari
import numpy as np
from magicgui import magic_factory
from micro_sam.sam_annotator import annotator_2d
from natsort import natsorted

from patho_sam.annotation import postprocess_instance_mask

DATA_DIR = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/eval_models/new_non_tumor_data"


class AnnotatorState:
    def __init__(self):
        self.inst_label_version = None


def get_grid(img_shape, cell_width=512) -> np.ndarray:
    """Add Grid for better overview in annotated regions"""
    return [
        np.array([[0, cell_width * i], [img_shape[0], cell_width * i]])
        for i in range(0, img_shape[1] // cell_width + 1)
    ] + [
        np.array([[cell_width * i, 0], [cell_width * i, img_shape[1]]])
        for i in range(0, img_shape[0] // cell_width + 1)
    ]


def save_to_h5(h5_path, save_key, data) -> None:
    with h5py.File(h5_path, "a") as f:
        if save_key in f:
            del f[save_key]
        f.create_dataset(name=save_key, data=data, compression="gzip", dtype=np.uint32)


def get_available_seg_versions(h5_path, prelim=False) -> list:
    with h5py.File(h5_path, "r") as f:
        if "inst_labels" not in f.keys():
            return []
        if prelim:
            return [key for key in f["inst_labels"].keys() if "prelim" in key]
        return list(f["inst_labels"].keys())


def start_interactive_annotator(h5_dir, embedding_dir, correction_version=None):
    h5_paths = natsorted(list(h5_dir.glob("*.h5")))

    for h5_path in h5_paths:
        img_name = h5_path.stem
        embedding_path = embedding_dir / img_name
        if not embedding_path.exists():
            raise ValueError("Embeddings are not precomputed!")
        with h5py.File(h5_path, "a") as f:
            image = f["img"][:].transpose(1, 2, 0)
            segmentation_result = postprocess_instance_mask(f["inst_pred"][:])
            # segmentation_result = None if "inst_labels" in f.keys() else postprocess_instance_mask(f["inst_pred"][:])
            if f"inst_labels/v_{correction_version}" in f or f"inst_labels/v_{int(correction_version) + 1}" in f:
                print(f"version {correction_version} already exists for {h5_path.name}")
                for prelim_version in [key for key in f["inst_labels"].keys() if "prelim" in key]:
                    print(f"Deleting inst_labels/{prelim_version}")
                    del f[f"inst_labels/{prelim_version}"]
                continue

        state = AnnotatorState()

        viewer = annotator_2d(
            image=image,
            embedding_path=embedding_path,
            segmentation_result=segmentation_result,
            model_type="vit_b_histopathology",
            tile_shape=(384, 384),
            halo=(64, 64),
            return_viewer=True,
            interpolation2d="linear",
            contrast_limits=(0, 255),
            rendering="mip",
        )
        if segmentation_result is not None:
            print("Loaded PathoSAM prediction.")
            state.inst_label_version = "v_0"

        viewer.add_shapes(
            get_grid(image.shape), shape_type="polygon", name=f"{img_name}_{state.inst_label_version}", edge_width=3
        )

        @magic_factory(
            call_button="Load previous instance label",
            version={"widget_type": "ComboBox", "choices": get_available_seg_versions(h5_path)},
        )
        def load_previous_annotation(viewer: "napari.Viewer", version: str = None):
            committed_objects = viewer.layers["committed_objects"]
            if version is None:
                print("No version selected")
                return
            with h5py.File(h5_path, "r") as f:
                previous_inst_label = f[f"inst_labels/{version}"][:]
            committed_objects.data = postprocess_instance_mask(previous_inst_label, area_threshold=160)
            print(f"Instance label {version} loaded.")
            grid_layer = viewer.layers[f"{img_name}_{state.inst_label_version}"]
            state.inst_label_version = version
            grid_layer.name = f"{img_name}_{state.inst_label_version}"

        @magic_factory(
            call_button="Apply postprocessing",
            overwrite_current={"label": "Overwrite commited objects"},
            min_area={"label": "Minimum object area"},
            intensity_threshold={"label": "Max. intensity threshold"},
        )
        def apply_postproc(
            viewer: "napari.Viewer",
            overwrite_current: bool = False,
            intensity_threshold: int = None,
            min_area: int = None,
        ):
            layer_name = "committed_objects"
            img_name = "image"
            img = viewer.layers[img_name].data
            current_seg = viewer.layers[layer_name].data

            postprocessed_seg = postprocess_instance_mask(
                segmentation=current_seg, image=img, intensity_threshold=intensity_threshold, area_threshold=min_area
            )
            if overwrite_current:
                viewer.layers[layer_name].data = postprocessed_seg
            else:
                viewer.add_labels(postprocessed_seg, name="Postprocessed")

        @magic_factory(
            call_button="Save preliminary annotation",
            save_to_version={"label": "Save to current version"},
            note={"label": "Specification", "widget_type": "LineEdit"},
        )
        def save_prelim_annotation(viewer: "napari.Viewer", note: str = "", save_to_version: bool = False):
            layer_name = "committed_objects"

            if layer_name not in viewer.layers:
                print("layer not found")
                return

            prelim_ann = viewer.layers[layer_name].data

            if save_to_version:
                save_key = "inst_labels/" + f"v_{state.inst_label_version.split('_')[1]}_prelim"
            else:
                save_key = "inst_labels/" + f"v_{str(int(state.inst_label_version.split('_')[1]) + 1)}_prelim"

            if note:
                save_key += f"_{note}"

            save_to_h5(h5_path, save_key, data=prelim_ann)

        @magic_factory(call_button="Save definite annotation", save_to_version={"label": "Save to current version"})
        def save_annotation(viewer: "napari.Viewer", save_to_version: bool = False):
            layer_name = "committed_objects"

            if layer_name not in viewer.layers:
                print("layer not found")
                return

            ann = viewer.layers[layer_name].data

            current_versions = get_available_seg_versions(h5_path)
            non_prelim = sorted([version for version in current_versions if "prelim" not in version])

            if non_prelim:
                most_recent_version = non_prelim[-1].split("_")[1]
            else:
                most_recent_version = "1"

            if save_to_version:
                save_key = "inst_labels/" + f"v_{most_recent_version}"
            else:
                save_key = "inst_labels/" + f"v_{str(int(most_recent_version) + 1)}"

            ann = postprocess_instance_mask(ann, verbose=True)

            save_to_h5(h5_path, save_key, data=ann)
            print(f"Saved to {h5_path} under {save_key}")

        viewer.window.add_dock_widget(save_prelim_annotation(), area="right")
        viewer.window.add_dock_widget(save_annotation(), area="right")
        viewer.window.add_dock_widget(load_previous_annotation(), area="right")
        viewer.window.add_dock_widget(apply_postproc(), area="right")

        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--correction", default=0, type=int)
    args = parser.parse_args()
    start_interactive_annotator(
        h5_dir=Path(args.data_dir) / "fixed_h5_files",
        embedding_dir=Path(args.data_dir) / "embeddings",
        correction_version=args.correction,
    )


if __name__ == "__main__":
    main()
