from pathlib import Path
import h5py
from tqdm import tqdm
import numpy as np
from natsort import natsorted
from micro_sam import util
from micro_sam.object_classification import compute_object_features

ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models"

predictor, _ = util.get_sam_model(
                    device="cpu", model_type="vit_b_histopathology",
                    checkpoint_path=None, decoder_path=None, return_state=True,
                    progress_bar_factory=None,
                )

for dir in Path(ROOT).glob("*non_tumor_data"):
    h5_files = natsorted((dir / "h5_files").glob("*.h5"))
    embedding_paths = natsorted((dir / "embeddings").glob("*"))
    if not len(h5_files) == len(embedding_paths):
        raise ValueError("Inconsistent input data!")
    for h5_file, embedding_path in tqdm(zip(h5_files, embedding_paths), total=len(h5_files)):
        with h5py.File(h5_file, 'a') as f:
            if "inst_label" not in f:
                continue
            else:
                inst_label = f["inst_label"][:]
                ids = np.unique(inst_label)
                unique_ids = len(ids[ids != 0])
                if "object_features" in f and "seg_ids" in f:
                    seg_ids = f["seg_ids"][:]
                    if unique_ids != seg_ids.shape[0]:
                        del f["seg_ids"]
                        del f["object_features"]
                    else:
                        continue
            if "img" not in f:
                print(f"missing image in {str(h5_file)}")
                continue
            img = f["img"][:]

            embeddings = util.precompute_image_embeddings(
                                    input_=img,
                                    predictor=predictor,
                                    ndim=2,
                                    tile_shape=(384, 384),
                                    halo=(64, 64),
                                    save_path=embedding_path
                    )
            seg_ids, features = compute_object_features(segmentation=inst_label,
                                                        image_embeddings=embeddings)
            f.create_dataset(name="object_features", data=features, compression="gzip")
            f.create_dataset(name="seg_ids", data=seg_ids, compression="gzip")
