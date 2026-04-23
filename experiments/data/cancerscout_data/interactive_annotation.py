import os
from magicgui import magic_factory
import napari
import numpy as np
from glob import glob
from natsort import natsorted
import argparse
import h5py
from micro_sam.sam_annotator import annotator_2d
from patho_sam.annotation import postprocess_instance_mask

DATA_DIR = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/new_tumor_data"


class AnnotatorState:
    def __init__(self):
        self.inst_label_version = None


def get_grid(img_shape, cell_width=512) -> np.ndarray:
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


def save_to_h5(h5_path, save_key, data) -> None:
    with h5py.File(h5_path, 'a') as f:
        if save_key in f:
            del f[save_key]
        f.create_dataset(name=save_key, data=data, compression="gzip", dtype=np.uint32)


def get_available_seg_versions(h5_path, prelim=False) -> list:
    with h5py.File(h5_path, 'a') as f:
        if prelim:
            return [key for key in f["inst_labels"].keys() if 'prelim' in key]
        return list(f["inst_labels"].keys())


def start_interactive_annotator(h5_dir, embedding_dir, correction_version=None):
    embedding_paths = natsorted(glob(os.path.join(embedding_dir, "*")))
    h5_paths = natsorted(glob(os.path.join(h5_dir, "*.h5")))

    for h5_path, embedding_path in zip(h5_paths, embedding_paths):
        img_name = os.path.basename(embedding_path)
        state = AnnotatorState()
        with h5py.File(h5_path, 'r') as f:
            image = f["img"][:]
            if f'inst_labels/v_{correction_version}' in f or f"inst_labels/v_{int(correction_version)+1}" in f:
                print(f"version {correction_version} already exists for {os.path.basename(h5_path)}")
                for prelim_version in [key for key in f["inst_labels"].keys() if 'prelim' in key]:
                    print(f"Deleting inst_labels/{prelim_version}")
                    del f[f'inst_labels/{prelim_version}']
                continue

        viewer = annotator_2d(
            image=image,
            embedding_path=embedding_path,
            model_type="vit_b_histopathology",
            tile_shape=(384, 384),
            halo=(64, 64),
            return_viewer=True,
        )

        viewer.add_shapes(get_grid(image.shape), shape_type="polygon", name=f"{img_name}_{state.inst_label_version}",
                          edge_width=3)

        @magic_factory(call_button="Load previous instance label",
                       version={"choices": get_available_seg_versions(h5_path)})
        def load_previous_annotation(viewer: "napari.Viewer", version: str = None):
            committed_objects = viewer.layers['committed_objects']
            if version is None:
                print("No version selected")
                return
            with h5py.File(h5_path, 'r') as f:
                previous_inst_label = f[f'inst_labels/{version}'][:]
            committed_objects.data = previous_inst_label
            print(f"Instance label {version} loaded.")
            grid_layer = viewer.layers[f"{img_name}_{state.inst_label_version}"]
            state.inst_label_version = version
            grid_layer.name = f"{img_name}_{state.inst_label_version}"

        @magic_factory(call_button="Save preliminary annotation", save_to_version={"label": "Save to current version"},
                       note={"label": "Specification", "widget_type": "LineEdit"})
        def save_prelim_annotation(viewer: "napari.Viewer", note: str = "", save_to_version: bool = False):
            layer_name = "committed_objects"

            if layer_name not in viewer.layers:
                print("layer not found")
                return

            prelim_ann = viewer.layers[layer_name].data

            if save_to_version:
                save_key = "inst_labels/" + f"v_{state.inst_label_version.split('_')[1]}_prelim"
            else:
                save_key = "inst_labels/" + f"v_{str(int(state.inst_label_version.split('_')[1])+1)}_prelim"

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

            if save_to_version:
                save_key = "inst_labels/" + f"v_{state.inst_label_version.split('_')[1]}"
            else:
                save_key = "inst_labels/" + f"v_{str(int(state.inst_label_version.split('_')[1])+1)}"

            ann = postprocess_instance_mask(ann, verbose=True)

            save_to_h5(h5_path, save_key, data=ann)
            print(f"Saved to {h5_path} under {save_key}")

        @magic_factory(call_button="Delete instance label",
                       version={"choices": lambda w: get_available_seg_versions(h5_path, prelim=True)})
        def delete_previous_annotation(viewer: "napari.Viewer", version=None):
            if version is None:
                print("No version selected")
                return
            with h5py.File(h5_path, 'a') as f:
                del f[f'inst_labels/{version}']
            print(f"Instance label {version} deleted.")

        viewer.window.add_dock_widget(save_prelim_annotation(), area="right")
        viewer.window.add_dock_widget(save_annotation(), area="right")
        viewer.window.add_dock_widget(load_previous_annotation(), area="right")
        viewer.window.add_dock_widget(delete_previous_annotation(), area="right")

        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--correction", default=0, type=int)
    args = parser.parse_args()
    start_interactive_annotator(
        h5_dir=os.path.join(args.data_dir, "new_h5_files"),
        embedding_dir=os.path.join(args.data_dir, "embeddings"),
        correction_version=args.correction,
    )


if __name__ == "__main__":
    main()
