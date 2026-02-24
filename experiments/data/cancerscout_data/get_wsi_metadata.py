import os
from glob import glob
import pyvips
import json
from tqdm import tqdm
import pandas as pd
import numpy as np

ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/"

if not os.path.exists("cancerscout_files_eval.json"):
    eval_dirs = glob(os.path.join(ROOT, "eval_models", "CancerScout_Lung", "*"))
    eval_dict = {}
    for eval_dir in eval_dirs:
        dir_name = os.path.basename(eval_dir)
        eval_dict[dir_name] = {"dims": [], "filesizes": [], "img_names": []}
        for image_path in tqdm(glob(os.path.join(eval_dir, "*.tiff"))):
            img = pyvips.Image.new_from_file(image_path, access="sequential")
            eval_dict[dir_name]["dims"].append([img.width, img.height])
            eval_dict[dir_name]["filesizes"].append(os.path.getsize(image_path) / 1e9)
            eval_dict[dir_name]["img_names"].append(os.path.basename(image_path.split(".")[0]))

    with open("cancerscout_files_eval.json", 'w') as f:
        json.dump(eval_dict, f, indent=2)

if not os.path.exists("cancerscout_files_train.json"):
    eval_dirs = glob(os.path.join(ROOT, "train_models", "CancerScout_Lung", "*"))
    eval_dict = {}
    for eval_dir in eval_dirs:
        dir_name = os.path.basename(eval_dir)
        eval_dict[dir_name] = {"dims": [], "filesizes": [], "img_names": []}
        for image_path in tqdm(glob(os.path.join(eval_dir, "*.tiff"))):
            img = pyvips.Image.new_from_file(image_path, access="sequential")
            eval_dict[dir_name]["dims"].append([img.width, img.height])
            eval_dict[dir_name]["filesizes"].append(os.path.getsize(image_path) / 1e9)
            eval_dict[dir_name]["img_names"].append(os.path.basename(image_path.split(".")[0]))

    with open("cancerscout_files_train.json", 'w') as f:
        json.dump(eval_dict, f, indent=2)


OUTPUT = "/user/titus.griebel/u23324/cancerscout_statistics"
with open("cancerscout_files_train.json", 'r') as f:
    train_dict = json.load(f)


# size_dict = {}
# dims_dict = {}
# img_names_dict = {}

# for mode, properties in train_dict.items():
#     size_dict[mode] = properties["filesizes"]
#     dims_dict[mode] = np.mean(np.array(properties["dims"]), axis=0).tolist()
#     img_names_dict[mode] = properties["img_names"]
# size_df = pd.DataFrame({k: pd.Series(v) for k, v in size_dict.items()})
# dims_df = pd.DataFrame({k: pd.Series(v) for k, v in dims_dict.items()})
# img_names_df = pd.DataFrame({k: pd.Series(v) for k, v in img_names_dict.items()})

# size_df.to_csv(os.path.join(OUTPUT, "file_sizes_train.csv"))
# dims_df.to_csv(os.path.join(OUTPUT, "wsi_dims_train.csv"))
# img_names_df.to_csv(os.path.join(OUTPUT, "img_names_train.csv"))


with open("cancerscout_files_eval.json", 'r') as f:
    eval_dict = json.load(f)

# size_dict = {}
# dims_dict = {}
# img_names_dict = {}

# for mode, properties in eval_dict.items():
#     size_dict[mode] = properties["filesizes"]
#     dims_dict[mode] = np.mean(np.array(properties["dims"]), axis=0).tolist()
#     img_names_dict[mode] = properties["img_names"]
# size_df = pd.DataFrame({k: pd.Series(v) for k, v in size_dict.items()})
# dims_df = pd.DataFrame({k: pd.Series(v) for k, v in dims_dict.items()})
# img_names_df = pd.DataFrame({k: pd.Series(v) for k, v in img_names_dict.items()})

# size_df.to_csv(os.path.join(OUTPUT, "file_sizes_eval.csv"))
# dims_df.to_csv(os.path.join(OUTPUT, "wsi_dims_eval.csv"))
# img_names_df.to_csv(os.path.join(OUTPUT, "img_names_eval.csv"))

for mode, properties in train_dict.items():

    file_size_avg = np.mean(train_dict[mode]["filesizes"])
    print(f"{file_size_avg} GB on average for mode {mode}")
