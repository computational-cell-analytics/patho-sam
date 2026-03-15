import pyvips
import os
from glob import glob
from tqdm import tqdm

pyvips.cache_set_max_mem(32 * 1024 * 1024 * 1024)  # 32 GB cache
pyvips.cache_set_max(10000)
# pyvips.concurrency_set(0)

ROOTS = [
    # "/mnt/ceph-hdd/cold/nim00020/hannibal_data/eval_models/CancerScout_Lung/new_non_tumor",
         "/mnt/ceph-hdd/cold/nim00020/hannibal_data/eval_models/CancerScout_Lung/new_tumor",
]


def transform_to_pyramid(image_path: str, output_dir):
    vips_img = pyvips.Image.new_from_file(image_path)

    output_path = os.path.join(output_dir, os.path.basename(image_path).replace(".tiff", "_pyramid.tiff"))
    if os.path.exists(output_path):
        return

    vips_img.tiffsave(
        output_path,
        tile=True,
        pyramid=True,
        bigtiff=True,
        compression="deflate",
        tile_width=512,
        tile_height=512,
        properties=False,
    )


def get_pyramid_tiffs(input_path, no_images):
    output_dir = os.path.join(os.path.dirname(input_path), f"pyramid_{os.path.basename(input_path)}")
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob(os.path.join(input_path, "*.tiff"))
    i = 0
    for image_path in tqdm(image_paths):
        if len(os.listdir(output_dir)) > no_images:
            break
        if i > no_images:
            return
        if (os.path.getsize(image_path) / 1e9) > 1.0:
            continue
        transform_to_pyramid(image_path, output_dir)
        i += 1


for root in ROOTS:
    get_pyramid_tiffs(root, no_images=6)
