from pathlib import Path

import h5py
from he_segmentation import get_model_and_params, infer_on_image
from tqdm import tqdm

model_root = "/user/titus.griebel/u23324/ignite-data-toolkit/data/models"

DATA_DIR = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/")

trainer, params = get_model_and_params(model_root)


h5_files = list(DATA_DIR.glob("*models/*_data/*fixed_h5_files/*.h5"))

for h5_file in tqdm(h5_files):
    with h5py.File(h5_file, "a") as f:
        if "ignite_pred" in f:
            # print(f"Pred already exists for {h5_file.name}")
            continue
        else:
            img = f["img"][:].transpose(1, 2, 0)
            ignite_pred = infer_on_image(img, trainer, params)
            f.create_dataset(name="ignite_pred", data=ignite_pred, compression="gzip")
