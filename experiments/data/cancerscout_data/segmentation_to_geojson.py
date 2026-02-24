import numpy as np
import tifffile
from skimage.measure import find_contours
import json
from tqdm import tqdm
import os
import argparse
from glob import glob
from shapely.geometry import Polygon, mapping


def tile_generator(img, tile_size=1024, overlap=0):
    """Yield tiles and their top-left coordinates."""
    h, w = img.shape
    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = min(y+tile_size, h), min(x+tile_size, w)
            yield img[y:y1, x:x1], (x, y)


def tile_to_polygons(tile, offset=(0, 0)):
    """Convert a label tile to polygons with offset."""
    polygons = []
    for label_id in np.unique(tile):
        if label_id == 0:
            continue
        mask = tile == label_id
        contours = find_contours(mask, 0.5)
        for contour in contours:
            coords = [(x + offset[0], y + offset[1]) for y, x in contour]
            polygons.append({
                "type": "Feature",
                "geometry": mapping(Polygon(coords)),
                "properties": {"classification": "Cell"}
            })
    return polygons


def segmentation_to_geojson(pred_path, tile_size=1024, overlap=128):
    mask = tifffile.imread(pred_path)
    tiff_name = os.path.splitext(pred_path)[0]
    if os.path.exists(f"{tiff_name}.geojson"):
        return

    annotations = []
    for tile, offset in tile_generator(mask, tile_size, overlap):
        annotations.extend(tile_to_polygons(tile, offset))

    geojson = {"type": "FeatureCollection", "features": annotations}
    with open(f"{tiff_name}.geojson", "w") as f:
        json.dump(geojson, f)


def convert_tiffs_from_dir(args):
    non_geojson_paths = [
        pred_path for pred_path in glob(os.path.join(args.input_dir, "*"))
        if not pred_path.endswith(".geojson")
    ]
    for non_geojson_path in tqdm(non_geojson_paths):
        segmentation_to_geojson(non_geojson_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", required=True, type=str)
    args = parser.parse_args()
    convert_tiffs_from_dir(args)


if __name__ == "__main__":
    main()