import os
import pyvips
from tqdm import tqdm
import imageio.v3 as imageio
import numpy as np
import zarr
import json
from glob import glob
from micro_sam.util import precompute_image_embeddings
from micro_sam.instance_segmentation import TiledAutomaticPromptGenerator, get_predictor_and_decoder

ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/"

JSON_PATHS = glob("/user/titus.griebel/u23324/patho-sam/experiments/data/cancerscout_data/*.json")
SQUARE_LENGTH = 5120


def compute_embeddings_for_roi(predictor, roi, tile_shape, halo, roi_embeddings_dir):
    return precompute_image_embeddings(predictor=predictor,
                                       input_=roi,
                                       tile_shape=tile_shape,
                                       halo=halo,
                                       save_path=roi_embeddings_dir,
                                       verbose=True,
                                       batch_size=12,
                                       ndim=2,
                                       )


def get_segmentation(generator, image, tile_shape, halo, embeddings=None, embedding_path=None):
    if embeddings is None:
        embeddings = zarr.open(embedding_path, mode='r')
    generator.initialize(image, image_embeddings=embeddings, tile_shape=tile_shape, halo=halo, verbose=True,
                         batch_size=12)
    segmentation = generator.generate(batch_size=32, optimize_memory=True)
    return segmentation


def process_selected_rois(img_type: str, coordinate_dict: dict, predictor, decoder, generator, split):
    tile_shape, halo = (384, 384), (64, 64)
    rois_dir = os.path.join(ROOT, f"{split}_models", f"rois_{img_type}")
    roi_images_dir = os.path.join(rois_dir, "images")
    rois_embeddings_dir = os.path.join(rois_dir, "embeddings")
    segmentation_dir = os.path.join(rois_dir, "segmentations")
    os.makedirs(rois_embeddings_dir, exist_ok=True)
    os.makedirs(roi_images_dir, exist_ok=True)
    os.makedirs(segmentation_dir, exist_ok=True)

    for img_name, roi_position in tqdm(coordinate_dict.items()):
        non_pyramid_path = os.path.join(ROOT, f"{split}_models", "CancerScout_Lung", img_type, img_name.replace("_pyramid.tiff", ".tiff"))
        roi_name = os.path.basename(non_pyramid_path.split(".")[0])
        if os.path.exists(os.path.join(segmentation_dir, f"{roi_name}.tiff")):
            continue
        embedding_dir = os.path.join(rois_embeddings_dir, roi_name)
        image = pyvips.Image.new_from_file(non_pyramid_path, access='sequential')
        patch = image.crop(roi_position[0], roi_position[1], SQUARE_LENGTH, SQUARE_LENGTH)
        patch_np = np.ndarray(buffer=patch.write_to_memory(),
                              dtype=np.uint8,
                              shape=[patch.height, patch.width, patch.bands])
        imageio.imwrite(os.path.join(roi_images_dir, f"roi_{roi_name}.tiff"), patch_np,  plugin="tifffile", compression="zlib")
        if os.path.exists(embedding_dir):
            if len(os.listdir(embedding_dir)) > 0:
                segmentation = get_segmentation(generator, patch_np, tile_shape, halo, embedding_path=embedding_dir)
                imageio.imwrite(os.path.join(segmentation_dir, f"{roi_name}.tiff"), segmentation)
                continue

        embeddings = compute_embeddings_for_roi(predictor=predictor,
                                                roi=patch_np,
                                                tile_shape=tile_shape,
                                                halo=halo,
                                                roi_embeddings_dir=embedding_dir
                                                )
        segmentation = get_segmentation(generator, patch_np, tile_shape, halo, embeddings=embeddings)
        imageio.imwrite(os.path.join(segmentation_dir, f"{roi_name}.tiff"), segmentation,  plugin="tifffile", compression="zlib")


def get_rois():
    predictor, decoder = get_predictor_and_decoder(model_type="vit_b_histopathology")
    generator = TiledAutomaticPromptGenerator(predictor, decoder)
    for json_path in JSON_PATHS:
        if os.path.basename(json_path).startswith("training"):
            split = "train"
        else:
            split = "eval"
        with open(json_path, 'r') as f:
            roi_dict = json.load(f)
        for img_type, coordinate_dict in roi_dict.items():
            print(f"Processing {img_type} images")
            process_selected_rois(img_type, coordinate_dict, predictor, decoder, generator, split)
            print(f"Finished processing {img_type} images")

get_rois()