from multiprocessing import cpu_count
from pathlib import Path

import h5py
from micro_sam.instance_segmentation import get_predictor_and_decoder
from micro_sam.util import precompute_image_embeddings
from tqdm import tqdm

from patho_sam.annotation import compute_object_features_parallel, postprocess_instance_mask

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/")

# all_h5_files = list(ROOT.glob("*models/*data/fixed_h5_files/*.h5"))
# only look at non tumor eval
all_h5_files = list(ROOT.glob("eval_models/new_non_tumor_data/fixed_h5_files/*.h5"))

predictor, decoder = get_predictor_and_decoder(model_type="vit_b_histopathology")

for h5_file in tqdm(all_h5_files):
    embedding_path = h5_file.parent.parent / "embeddings" / h5_file.stem
    if not embedding_path.exists():
        print(f"No precomputed embeddings at {embedding_path}")
        continue

    with h5py.File(h5_file, "a") as f:
        if "inst_labels/v_2" in f:
            img = f["img"][:].transpose(1, 2, 0)
            embeddings = precompute_image_embeddings(
                input_=img, predictor=predictor, tile_shape=(384, 384), ndim=2, halo=(64, 64), save_path=embedding_path
            )
            if "all_features" in f:
                del f["all_features"]
                del f["all_seg_ids"]
            seg = f["inst_labels/v_2"][:]
            seg = postprocess_instance_mask(seg, intensity_threshold=200, area_threshold=50, image=img)
            seg_ids, features = compute_object_features_parallel(seg, embeddings, n_workers=max(cpu_count() - 1, 1))
            f.create_dataset("all_features", data=features, compression="gzip")
            f.create_dataset("all_seg_ids", data=seg_ids, compression="gzip")
            del f["inst_labels/v_2_pproc"]
            f.create_dataset("inst_labels/v_2_pproc", data=seg, compression="gzip")
        else:
            print(f"no v_2 label for {h5_file}")
