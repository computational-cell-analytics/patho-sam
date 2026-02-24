import os
import slideio
from tqdm import tqdm
from glob import glob
import imageio.v3 as imageio

ROOT = '/mnt/ceph-hdd/cold/nim00020/hannibal_data/eval_models/CancerScout_Lung/new_tumor_preds'
IMG_ROOT = '/mnt/ceph-hdd/cold/nim00020/hannibal_data/eval_models/CancerScout_Lung/new_tumor'


def save_roi_image(image_path, roi, output_path):
    slide = slideio.open_slide(image_path)
    scene = slide.get_scene(0)
    input_array = scene.read_block(rect=roi, size=(0, 0))
    imageio.imwrite(output_path, input_array)


def get_roi_images():
    for pred_path in glob(os.path.join(ROOT, "*instances.tif")):
        image_path = glob(os.path.join(IMG_ROOT, f"*{os.path.basename(pred_path).split("_")[0]}*"))
        assert len(image_path) == 1, "Unexpectedly many matches for searched original image"
        roi = "_".join(os.path.basename(pred_path).split("_")[5:7])
        x_part, y_part = roi[1:].split("_Y")

        x_start, x_end = map(int, x_part.split("-"))
        y_start, y_end = map(int, y_part.split("-"))

        rect = (
            x_start,
            y_start,
            x_end - x_start,
            y_end - y_start,
        )
        output_path = pred_path.replace("instances.tif", "raw_image.tif")

        save_roi_image(image_path=image_path[0], roi=rect, output_path=output_path)


def get_roi_examples(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rois = [(20000, 20000, 10000, 10000), (20000, 30000, 10000, 10000), (30000, 20000, 10000, 10000)]
    for image_path in tqdm(glob(os.path.join(IMG_ROOT, "*.tiff"))[30:60]):
        for idx, roi in enumerate(rois):
            output_path = os.path.join(output_dir, os.path.basename(image_path).replace(".tiff", f"roi_{idx}.tiff"))
            if os.path.exists(output_path):
                continue
            save_roi_image(image_path, roi, output_path=output_path)


get_roi_examples("/mnt/ceph-hdd/cold/nim00020/hannibal_data/eval_models/CancerScout_Lung/new_tumor_rois_2")
# get_roi_images()