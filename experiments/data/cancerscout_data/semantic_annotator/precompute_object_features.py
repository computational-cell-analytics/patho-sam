import os
import numpy as np
import imageio.v3 as imageio
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from micro_sam.object_classification import compute_object_features
from micro_sam import util

INPUT_ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/"


def precompute_features(predictor, feature_cache_dir, embedding_paths, annotation_paths, image_paths):

    if not len(embedding_paths) == len(annotation_paths) == len(image_paths):
        print( f"Inconsistent input: {len(image_paths)} images, {len(annotation_paths)} segs, {len(embedding_paths)} embeddings")
        return

    os.makedirs(feature_cache_dir, exist_ok=True)

    for img_path, embedding_path, annotation_path in tqdm(zip(image_paths, embedding_paths, annotation_paths),
                                                        total=len(embedding_paths)):
        img_name = os.path.basename(embedding_path)
        features_cache_path = os.path.join(feature_cache_dir, f"{img_name}_features.npy")
        if os.path.exists(features_cache_path):
            print(f"Features for img {img_name} already cached")
            continue
        embeddings = util.precompute_image_embeddings(
            input_=imageio.imread(img_path),
            predictor=predictor,
            ndim=2,
            tile_shape=(384, 384),
            halo=(64, 64), 
            save_path=embedding_path
        )
        seg_ids, features = compute_object_features(segmentation=imageio.imread(annotation_path),
                                                    image_embeddings=embeddings)
        np.save(os.path.join(feature_cache_dir, f"{img_name}_seg_ids.npy"), seg_ids)
        np.save(features_cache_path, features)


def precompute_for_all():
    predictor, _ = util.get_sam_model(
                    device="cpu", model_type="vit_b_histopathology",
                    checkpoint_path=None, decoder_path=None, return_state=True,
                    progress_bar_factory=None,
                )
    for roi_dir in glob(os.path.join(INPUT_ROOT, "*_models", "rois_*")):
        print(roi_dir)
        embedding_paths = natsorted(glob(os.path.join(roi_dir, "embeddings", "*")))
        annotation_paths = natsorted(glob(os.path.join(roi_dir, "annotations", "*label*")))
        image_paths = natsorted(glob(os.path.join(roi_dir, "images", "*")))
        feature_cache_dir = os.path.join(roi_dir, "cached_features")

        precompute_features(predictor=predictor,
                            feature_cache_dir=feature_cache_dir,
                            image_paths=image_paths,
                            annotation_paths=annotation_paths,
                            embedding_paths=embedding_paths)

precompute_for_all()