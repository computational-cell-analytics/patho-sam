import os
import shutil
import subprocess
from glob import glob
import tifffile as tiff
from natsort import natsorted
from scipy.io import loadmat
from tqdm import tqdm
import imageio.v3 as imageio
import cv2
import json
import numpy as np

def json_to_tiff(path):
    label_json_paths = [p for p in natsorted(glob(os.path.join(path, "json", "*.json")))]
    img_shape = (512, 512)
    os.makedirs(os.path.join(path, "semantic"), exist_ok=True)
    for mpath in tqdm(label_json_paths, desc="Postprocessing labels"):
        label_path = os.path.join(path, "semantic", os.path.basename(mpath.replace(".json", ".tiff")))
        with open(mpath, 'r') as file:
            data = json.load(file)
            pred_class_map = np.zeros(img_shape, dtype=np.int32)
            for id, cell_data in enumerate(data['nuc'].items(), start=1):
                cell_data = cell_data[1]
                contour = np.array(cell_data["contour"]) 
                contour[:, 0] = np.clip(contour[:, 0], 0, img_shape[1])
                contour[:, 1] = np.clip(contour[:, 1], 0, img_shape[0])
                contour = contour.reshape((-1, 1, 2))
                cell_type = cell_data["type"]
                contour = np.vstack((contour, [contour[0]]))
                contour = contour.astype(np.int32)
                cv2.fillPoly(pred_class_map, [contour], cell_type)        
        imageio.imwrite(label_path, pred_class_map)


def mat_to_tiff(path):
    os.makedirs(os.path.join(path, "instance"), exist_ok=True)
    label_mat_paths = [p for p in natsorted(glob(os.path.join(path, "mat", "*.mat")))]
    for mpath in tqdm(label_mat_paths, desc="Postprocessing predictions"):
        label_path = os.path.join(path, "instance", os.path.basename(mpath.replace(".mat", ".tiff")))
        label = loadmat(mpath)["inst_type"]
        tiff.imwrite(label_path, label)


def run_inference(model_dir, input_dir, output_dir, type_info_path):
    for dataset in ["pannuke"]:
        for checkpoint in ["pannuke"]:
            output_path = os.path.join(output_dir, "inference", dataset, checkpoint)
            input_path = os.path.join(input_dir, dataset, "loaded_testset", "eval_split", "test_images")
            if os.path.exists(os.path.join(output_dir, "results", dataset, checkpoint, 'ais_result.csv')):
                print(f"Inference with HoVerNet model (type: {checkpoint}) on {dataset} dataset already done")
                continue
            os.makedirs(output_path, exist_ok=True)
            if checkpoint in ["consep", "cpm17", "kumar"]:
                model_mode = "original"
                model_path = os.path.join(
                    model_dir, "checkpoints", f"hovernet_original_{checkpoint}_notype_tf2pytorch.tar"
                )
                nr_types = 0
            else:
                model_mode = "fast"

                model_path = os.path.join(model_dir, "checkpoints", f"hovernet_fast_{checkpoint}_type_tf2pytorch.tar")
                if checkpoint == "pannuke":
                    nr_types = 6
                else:
                    nr_types = 5

            args = [
                "--nr_types",
                f"{nr_types}",
                "--type_info_path",
                "/user/titus.griebel/u12649/hover_net/type_info.json",
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
                # "--save_raw_map",
            ]

            command = ["python3", "/user/titus.griebel/u12649/hover_net/run_infer.py"] + args
            print(f"Running inference with HoVerNet {checkpoint} model on {dataset} dataset...")

            subprocess.run(command)
            mat_to_tiff(output_path)
            json_to_tiff(output_path)
            shutil.rmtree(os.path.join(output_path, "json"))
            shutil.rmtree(os.path.join(output_path, "mat"))
            shutil.rmtree(os.path.join(output_path, "overlay"))
            print(f"Inference on {dataset} dataset with the HoVerNet {checkpoint} model successfully completed")


run_inference(
    model_dir="/mnt/lustre-grete/usr/u12649/models/hovernet",
    input_dir="/mnt/lustre-grete/usr/u12649/data/semantic_data",
    output_dir="/mnt/lustre-grete/usr/u12649/models/hovernet_types",
    type_info_path="/user/titus.griebel/u12649/hover_net/type_info.json",
)
