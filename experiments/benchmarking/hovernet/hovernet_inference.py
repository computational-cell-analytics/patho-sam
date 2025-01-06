import os
import shutil
import subprocess
from glob import glob

import imageio
from natsort import natsorted
from scipy.io import loadmat
from tqdm import tqdm

DATASETS = [
    "consep",
    "cpm15",
    "cpm17",
    "cryonuseg",
    "lizard",
    "lynsec_he",
    "lynsec_ihc",
    "monusac",
    "monuseg",
    "nuclick",
    "nuinsseg",
    "pannuke",
    "puma",
    "srsanet",
    "tnbc",
]


def mat_to_tiff(path):
    label_mat_paths = [p for p in natsorted(glob(os.path.join(path, "*.mat")))]
    label_paths = []
    for mpath in tqdm(label_mat_paths, desc="Preprocessing labels"):
        label_path = mpath.replace(".mat", "_instance_labels.tiff")
        label_paths.append(label_path)
        if os.path.exists(label_path):
            continue
        label = loadmat(mpath)["inst_map"]
        imageio.imwrite(label_path, label)
        os.remove(mpath)


def run_inference(model_dir, input_dir, output_dir, type_info_path):
    for dataset in DATASETS:
        for model in ["consep", "cpm17", "kumar", "pannuke", "monusac"]:
            output_path = os.path.join(output_dir, dataset, model)
            input_path = os.path.join(input_dir, dataset, "loaded_testset", "eval_split", "test_images")
            if os.path.exists(output_path):
                continue
            os.makedirs(output_path, exist_ok=True)
            if model in ["consep", "cpm17", "kumar"]:
                model_mode = "original"
                model_path = os.path.join(model_dir, f"hovernet_original_{model}_notype_tf2pytorch.tar")
                nr_types = 0
                type_info = ""
            else:
                model_mode = "fast"

                model_path = os.path.join(model_dir, f"hovernet_fast_{model}_type_tf2pytorch.tar")
                type_info = type_info_path
                if model == "pannuke":
                    nr_types = 6
                else:
                    nr_types = 5

            args = [
                "--nr_types",
                f"{nr_types}",
                "--type_info_path",
                f"{type_info}",
                "--model_mode",
                f"{model_mode}",
                "--model_path",
                f"{model_path}",
                "--nr_inference_workers",
                "2",
                "--nr_post_proc_worker",
                "0",
                "tile",
                "--input_dir",
                f"{input_path}",
                "--output_dir",
                f"{output_path}",
                "--save_raw_map",
            ]

            command = ["python3", "/user/titus.griebel/u12649/hover_net/run_infer.py"] + args
            print(f"Running inference with HoVerNet {model} model on {dataset} dataset...")

            subprocess.run(command)
            mat_to_tiff(os.path.join(output_path, "mat"))
            shutil.rmtree(os.path.join(output_path, "json"))
            shutil.rmtree(os.path.join(output_path, "overlay"))
            print(f"Inference on {dataset} dataset with the HoVerNet {model} model successfully completed")


run_inference(
    model_dir="/mnt/lustre-grete/usr/u12649/models/hovernet/checkpoints",
    input_dir="/mnt/lustre-grete/usr/u12649/data/final_test",
    output_dir="/mnt/lustre-grete/usr/u12649/models/hovernet/inference",
    type_info_path="/user/titus.griebel/u12649/hover_net/type_info.json",
)
