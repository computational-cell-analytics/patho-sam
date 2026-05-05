from multiprocessing import Pool, cpu_count
from typing import Tuple

import numpy as np
from micro_sam.object_classification import _compute_object_features_impl, _create_seg_and_embed_generator
from tqdm import tqdm


def _worker(args):
    seg, embeds, resize_embedding_shape = args
    seg_ids, features = _compute_object_features_impl(embeds, seg, resize_embedding_shape)
    return seg_ids, features


def compute_object_features_parallel(
    segmentation, image_embeddings, resize_embedding_shape: Tuple[int, int] = (256, 256), n_workers=None, verbose=True
):

    is_tiled = image_embeddings["input_size"] is None

    is_3d = segmentation.ndim == 3

    # If we have simple embeddings, i.e. 2d without tiling, then we can directly compute the features.
    if not is_tiled and not is_3d:
        embeddings = image_embeddings["features"].squeeze()
        return _compute_object_features_impl(embeddings, segmentation, resize_embedding_shape)
    if n_workers is None:
        n_workers = max(1, cpu_count() - 3)

    # generator for segmentation + embeddings
    seg_embed_generator, n_gen = _create_seg_and_embed_generator(
        segmentation, image_embeddings, is_tiled=is_tiled, is_3d=is_3d
    )

    # aggregation containers
    feature_sums = {}  # seg_id -> sum(weighted features)
    size_sums = {}  # seg_id -> total size

    def task_iter():
        for seg, embeds in seg_embed_generator():
            yield (seg, embeds, resize_embedding_shape)

    with Pool(n_workers) as pool:
        for seg_ids, feats in tqdm(
            pool.imap(_worker, task_iter()), total=n_gen, disable=not verbose, desc="Compute object features"
        ):
            seg_ids = seg_ids.tolist()

            for i, seg_id in enumerate(seg_ids):
                size = feats[i, 0]
                vec = feats[i, 1:]

                if seg_id not in feature_sums:
                    feature_sums[seg_id] = vec * size
                    size_sums[seg_id] = size
                else:
                    feature_sums[seg_id] += vec * size
                    size_sums[seg_id] += size

    # finalize
    seg_ids_sorted = sorted(feature_sums.keys())
    n_features = feats.shape[1]

    features = np.zeros((len(seg_ids_sorted), n_features), dtype="float32")

    for i, seg_id in enumerate(seg_ids_sorted):
        total_size = size_sums[seg_id]
        features[i, 0] = total_size
        features[i, 1:] = feature_sums[seg_id] / total_size

    return np.array(seg_ids_sorted), features
