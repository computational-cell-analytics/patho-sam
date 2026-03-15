import imageio.v3 as imageio
import os
from magicgui import magicgui
import napari
from glob import glob
from natsort import natsorted
import pandas as pd
import argparse
from micro_sam.sam_annotator import annotator_2d


def start_interactive_annotator(args):
    split = args.split + "_models"
    annotation_dir = os.path.join(args.root_dir, split, "rois_" + args.type, "annotations")
    os.makedirs(annotation_dir, exist_ok=True)
    embedding_paths = os.path.join(args.root_dir, split, "rois_" + args.type, "embeddings")
    image_paths = natsorted(glob(os.path.join(args.root_dir, split, "rois_" + args.type, "images", "*.tiff")))
    pred_dir = os.path.join(args.root_dir, split, "rois_" + args.type, "segmentations")
    correction_version = args.correction

    for image_path in image_paths:
        img_name = os.path.basename(image_path).strip(".tiff")[4:]
        short_img_name = img_name.split("-")[1] + "-" + "-".join(img_name.split("-")[-2:])
        annotation_path = os.path.join(annotation_dir, img_name + f"_label_{correction_version}.tiff")
        previous_version = annotation_path.replace(f"{correction_version}.tiff", f"{correction_version-1}.tiff")

        # In case the annotation for the current annotation version already exists, skip annotation process.
        if os.path.exists(annotation_path):
            print(f"Correction version {correction_version} already exists for {img_name}")
            continue
        else:
            # Check if lower version label exists
            if os.path.exists(previous_version):
                segmentation_result = imageio.imread(previous_version)
                grid_label = f"{short_img_name}-version_{correction_version - 1}"
                print(f"Successfully loaded annotation version {correction_version-1}")
            else:
                prelim_annotations = natsorted(glob(os.path.join(annotation_dir, f"*{img_name}*")))
                if len(prelim_annotations) > 0:
                    grid_label = short_img_name + "-" + "".join(prelim_annotations[0].strip("_prelim.tiff").split("_")[-1])
                    segmentation_result = imageio.imread(prelim_annotations[0])
                    print("Successfully loaded preliminary annotation")
                else:
                    pred_path = os.path.join(pred_dir, os.path.basename(image_path)[4:])
                    segmentation_result = imageio.imread(pred_path)
                    print("Loaded untouched PathoSAM pred")
                    grid_label = short_img_name + "-fresh"

        embedding_path = os.path.join(embedding_paths, os.path.basename(image_path.strip(".tiff"))[4:])

        image = imageio.imread(image_path)

        viewer = annotator_2d(
            image=image,
            embedding_path=embedding_path,
            segmentation_result=segmentation_result,
            model_type="vit_b_histopathology",
            tile_shape=(384, 384),
            halo=(64, 64),
            return_viewer=True,
        )

        if args.grid_path is not None:
            df = pd.read_csv(args.grid_path)
            shapes = []
            for shape_id, group in df.groupby("index"):
                coords = group[["axis-0", "axis-1"]].values
                shapes.append(coords)

            viewer.add_shapes(shapes, shape_type="polygon", name=grid_label, edge_width=3)

        @magicgui(call_button="Save preliminary annotation", note={"label": "Specification", "widget_type": "LineEdit"})
        def save_prelim_annotation(viewer: "napari.Viewer", note: str = ""):
            layer_name = "committed_objects"

            if layer_name not in viewer.layers:
                print("layer not found")
                return

            ann = viewer.layers[layer_name].data
            prelim_path = annotation_path.replace(f"_label_{correction_version}.tiff", f"_{note}_prelim_{correction_version}.tiff")
            imageio.imwrite(annotation_path.replace(f"_label_{correction_version}.tiff", f"_{note}_prelim_{correction_version}.tiff"), ann,  plugin="tifffile", compression="zlib")
            prelim_annotations = glob(os.path.join(annotation_dir, f"*{img_name}_*_prelim*"))
            for prelim_annotation in prelim_annotations:
                if not prelim_annotation == prelim_path:
                    os.remove(prelim_annotation)

        @magicgui(call_button="Save definite annotation")
        def save_annotation(viewer: "napari.Viewer"):
            layer_name = "committed_objects"

            if layer_name not in viewer.layers:
                print("layer not found")
                return

            prelim_ann = viewer.layers[layer_name].data
            imageio.imwrite(annotation_path, prelim_ann, plugin="tifffile", compression="zlib")
            prelim_annotations = glob(os.path.join(annotation_dir, f"*{img_name}_*_prelim*"))
            for prelim_annotation in prelim_annotations:
                os.remove(prelim_annotation)

        viewer.window.add_dock_widget(save_prelim_annotation, area="right")
        viewer.window.add_dock_widget(save_annotation, area="right")
        napari.run()
    print(f"Annotation round {correction_version} for {args.type} in {split} split completed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="/mnt/ceph-hdd/cold/nim00020/hannibal_data/")
    parser.add_argument("--grid_path", default="/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/rois_new_tumor/grid.csv")
    parser.add_argument("--correction", default=0, type=int)
    parser.add_argument("--type", default="new_tumor")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    start_interactive_annotator(args)


if __name__ == "__main__":
    main()
