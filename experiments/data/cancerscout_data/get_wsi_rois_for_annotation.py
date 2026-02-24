import os
import pyvips
from tqdm import tqdm
import imageio.v3 as imageio
import numpy as np
import zarr
from micro_sam.util import precompute_image_embeddings
from micro_sam.instance_segmentation import TiledAutomaticPromptGenerator, get_predictor_and_decoder

ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/CancerScout_Lung/new_tumor"

SQUARE_LENGTH = 5120

ROI_DICT = {"A2020-001087_1-1-1_HE-2021-09-07T14-36-30_pyramid.tiff": (42220, 12265),
            "A2020-001011_1-1-1_HE-2021-10-08T10-49-32_pyramid.tiff": (22680, 66460),
            "A2020-001021_1-1-1_HE-2021-10-08T09-25-14_pyramid.tiff": (79800, 45350),
            "A2020-001030_1-1-1_HE-2021-10-08T09-44-52_pyramid.tiff": (47300, 7000),
            "A2020-001040_1-1-1_HE-2021-10-08T15-19-09_pyramid.tiff": (20300, 41500),
            "A2020-001051_1-1-1_HE-2021-09-07T13-33-49_pyramid.tiff": (25300, 36774),
            "A2020-001059_1-1-1_HE-2021-10-08T14-24-51_pyramid.tiff": (34130, 40000),
            "A2020-001073_1-1-1_HE-2021-09-08T17-29-50_pyramid.tiff": (40000, 55900),
            "A2020-001093_1-1-1_HE-2021-09-07T14-48-48_pyramid.tiff": (18200, 38500),
            "A2020-001106_1-1-1_HE-2021-09-07T15-15-18_pyramid.tiff": (38355, 15820),
            }


def compute_embeddings_for_roi(predictor, roi, tile_shape, halo, roi_embeddings_dir):
    precompute_image_embeddings(predictor=predictor,
                                input_=roi,
                                tile_shape=tile_shape,
                                halo=halo,
                                save_path=roi_embeddings_dir,
                                verbose=True,
                                batch_size=12,
                                ndim=2,
                                )


def get_segmentation(generator, image, embedding_path, tile_shape, halo):
    embeddings = zarr.open(embedding_path, mode='r')
    generator.initialize(image, image_embeddings=embeddings, tile_shape=tile_shape, halo=halo, verbose=True,
                         batch_size=12)
    segmentation = generator.generate(batch_size=32, optimize_memory=True)
    return segmentation


def process_selected_rois():
    tile_shape, halo = (384, 384), (64, 64)
    predictor, decoder = get_predictor_and_decoder(model_type="vit_b_histopathology")
    rois_dir = os.path.join(os.path.dirname(ROOT), f"rois_{os.path.basename(ROOT)}")
    roi_images_dir = os.path.join(rois_dir, "images")
    rois_embeddings_dir = os.path.join(rois_dir, "embeddings")
    segmentation_dir = os.path.join(rois_dir, "segmentations")
    os.makedirs(rois_embeddings_dir, exist_ok=True)
    os.makedirs(roi_images_dir, exist_ok=True)
    os.makedirs(segmentation_dir, exist_ok=True)

    generator = TiledAutomaticPromptGenerator(predictor, decoder)

    for img_name, roi_position in tqdm(ROI_DICT.items()):
        non_pyramid_path = os.path.join(ROOT, img_name.replace("_pyramid.tiff", ".tiff"))
        roi_name = os.path.basename(non_pyramid_path.split(".")[0])
        embedding_dir = os.path.join(rois_embeddings_dir, roi_name)

        image = pyvips.Image.new_from_file(non_pyramid_path, access='sequential')
        patch = image.crop(roi_position[0], roi_position[1], SQUARE_LENGTH, SQUARE_LENGTH)
        patch_np = np.ndarray(buffer=patch.write_to_memory(),
                              dtype=np.uint8,
                              shape=[patch.height, patch.width, patch.bands])
        if os.path.exists(embedding_dir):
            if len(os.listdir(embedding_dir)) > 0:
                segmentation = get_segmentation(generator, patch_np, embedding_dir, tile_shape, halo)
                imageio.imwrite(os.path.join(segmentation_dir, f"{roi_name}.tiff"), segmentation)
                continue
        imageio.imwrite(os.path.join(roi_images_dir, f"roi_{roi_name}.tiff"), patch_np)

        compute_embeddings_for_roi(predictor=predictor,
                                   roi=patch_np,
                                   tile_shape=tile_shape,
                                   halo=halo,
                                   roi_embeddings_dir=embedding_dir
                                   )


process_selected_rois()
