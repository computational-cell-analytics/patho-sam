import json
from pathlib import Path

import pyvips
from tqdm import tqdm

pyvips.cache_set_max_mem(32 * 1024 * 1024 * 1024)  # 32 GB cache
pyvips.cache_set_max(10000)

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data")

reject_samples = {
    "eval": ["A2020-001296_1-1-1_HE-2021-10-08T16-31-57", "A2020-001672_1-1-1_HE-2021-09-29T07-59-26"],
    "train": [],
}


def transform_to_pyramid(image_path: Path, output_dir: Path):
    image_path = image_path.with_name(image_path.name.replace("_pyramid.tiff", ".tiff"))
    vips_img = pyvips.Image.new_from_file(image_path)

    output_path = output_dir / (image_path.stem + "_pyramid.tiff")
    print(f"Converting {image_path.stem}")

    vips_img.tiffsave(
        output_path,
        tile=True,
        pyramid=True,
        bigtiff=True,
        strip=True,
        compression="jpeg",
        tile_width=512,
        tile_height=512,
        properties=False,
        Q=90,
    )
    if not output_path.is_file():
        raise FileNotFoundError
    print(f"{image_path.name} saved to {output_path}")


def get_pyramid_tiffs(input_path: Path, entity, split):
    img_dir = input_path / f"{split}_models" / "CancerScout_Lung" / entity
    output_dir = input_path / f"{split}_models" / "CancerScout_Lung" / f"pyramid_{entity}"
    output_dir.mkdir(exist_ok=True)
    json_path = input_path / f"{split}_models" / f"{split}_rois.json"

    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)

    dict_entries_to_delete = []
    images_to_predict = [
        (img_dir / sample.replace("_pyramid.tiff", ".tiff"))
        for sample in data[entity].keys()
        if not (output_dir / sample).exists()
    ]

    images_without_coords = [key for key, coords in data[entity].items() if len(coords) == 0]

    for key in data[entity].keys():
        if key.strip("_pyramid.tiff") in reject_samples[split]:
            dict_entries_to_delete.append(key)

    for entry in dict_entries_to_delete:
        print(f"removed {entry} from split")
        del data[entity][entry]

    for image_path in tqdm(images_to_predict):
        transform_to_pyramid(image_path, output_dir)

    with open(ROOT / "pyramid_tiffs_to_download.json", "w") as f:
        data = {"samples": [str(p) for p in images_without_coords]}
        json.dump(data, f, indent=4)


get_pyramid_tiffs(ROOT, entity="new_tumor", split="eval")
