from pathlib import Path

import imageio.v3 as imageio
import napari
from magicgui import magic_factory
from natsort import natsorted
from preprocess_train_data import pad_to_multiple

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/ignite/data")


def correct_labels(path: Path):
    corrected_annotations_dir = path / "corrected_tissue_annotations" / "annotations" / "heww"
    corrected_annotations_dir.mkdir(exist_ok=True, parents=True)
    corrected_annotation_names = [
        p.name for p in corrected_annotations_dir.glob("*.png") if not p.name.endswith("_context.png")
    ]
    image_paths = natsorted(
        [
            p
            for p in (path / "images" / "images" / "he").glob("*.png")
            if p.name not in corrected_annotation_names and not p.name.endswith("context.png")
        ]
    )
    label_paths = natsorted(
        [path / "corrected_tissue_annotations" / "annotations" / "he" / p.name for p in image_paths]
    )
    # for img_path, label_path in tqdm(zip(image_paths, label_paths)):
    #     # img = imageio.imread(img_path)
    #     label = imageio.imread(label_path)
    #     label = 1 - label
    #     imageio.imwrite(corrected_annotations_dir / label_path.name, label)
    #     continue
    # label_remapped = remap_labels(label, "ignite")
    print(len(label_paths), len(image_paths))
    pairs = list(zip(image_paths, label_paths))
    idx = 0
    viewer = napari.Viewer()

    def load_sample(i):
        img_path, label_path = pairs[i]

        img = imageio.imread(img_path)
        label = imageio.imread(label_path)
        img, label_remapped = pad_to_multiple(img, label)

        if "img" not in viewer.layers:
            viewer.add_image(img, name="img")
            # viewer.add_labels(label, name="original_label")
            viewer.add_labels(label_remapped, name="remapped_label")
        else:
            viewer.layers["img"].data = img
            # viewer.layers["original_label"].data = label
            viewer.layers["remapped_label"].data = label_remapped

        return img_path

    current_img_path = load_sample(idx)
    # viewer.add_labels(label, name="original_label")
    # viewer.add_image(img, name="img")
    # viewer.add_labels(label_remapped, name="remapped_label")

    @magic_factory(call_button="Save preliminary annotation")
    def save_corrected_label(viewer: "napari.Viewer"):
        nonlocal idx, current_img_path

        corrected_label = viewer.layers["remapped_label"].data

        # imageio.imwrite(corrected_annotations_dir / current_img_path.name, corrected_label, compress_level=6)

        idx += 1

        if idx >= len(pairs):
            viewer.close()
            return

        current_img_path = load_sample(idx)

    viewer.window.add_dock_widget(save_corrected_label(), area="right")

    napari.run()


correct_labels(ROOT)
