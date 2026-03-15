from thesis_figures.crop_magnify import get_fancy_magnified_crop
from thesis_figures.methods.datasets.create_mask_overlay_figs import generate_annotation_visualization_fig, get_plot
import imageio.v3 as imageio
import os
from tqdm import tqdm
from glob import glob

# img_names = [
#              "A2020-001059_1-1-1_HE-2021-10-08T14-24-51_label.tiff",
#              "A2020-001093_1-1-1_HE-2021-09-07T14-48-48_label.tiff",
#             #  "A2020-001021_1-1-1_HE-2021-10-08T09-25-14_label.tiff"
#             "A2020-001040_1-1-1_HE-2021-10-08T15-19-09_label.tiff",
#              ]
img_dir = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/rois_new_tumor/images"
annotation_dir = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/rois_new_tumor/annotations"
output_dir = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/visualization_figs"

with_labels = False
img_names = [os.path.basename(path) for path in glob(os.path.join(annotation_dir, "*_label.tiff"))]

breakpoint()


for img_name in tqdm(img_names):
    img_path = os.path.join(img_dir, f"roi_{img_name.replace("_label.tiff", ".tiff")}")
    annotation_path = os.path.join(annotation_dir, img_name)
    img = imageio.imread(img_path)
    label = imageio.imread(annotation_path)

    if True:

        overlay_path = os.path.join(output_dir, os.path.basename(img_path).replace(".tiff", "_overlay.tiff"))
        if not os.path.exists(overlay_path):
            overlay = generate_annotation_visualization_fig(
                img, label, outline_width=0
            )
            print("label outlines generated")
            imageio.imwrite(overlay_path, overlay)
        else:
            overlay = imageio.imread(overlay_path)

        get_plot(overlay,
                 label=label,
                 output_path=os.path.join(output_dir, os.path.basename(img_path).replace(".tiff", "_overaly.tiff")),
                 alpha=0.3)

        print("overlay plot generated")

        get_fancy_magnified_crop(
            image_path=os.path.join(output_dir, os.path.basename(img_path).replace(".tiff", "_overaly.tiff")),
            output_path=os.path.join(output_dir, os.path.basename(img_path).replace(".tiff", "_crop_magnified.png")),
            rectangle_to_zoom=(2206, 808, 1000, 1000),
            zf=5
        )

    get_fancy_magnified_crop(
        image_path=img_path,
        output_path=os.path.join(output_dir, os.path.basename(img_path).replace(".tiff", "_crop_magnified_without_labels.png")),
        rectangle_to_zoom=(2206, 808, 1000, 1000),
        zf=5
    )

