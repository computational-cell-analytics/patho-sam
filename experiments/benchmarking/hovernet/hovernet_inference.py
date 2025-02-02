import os
import shutil
import argparse
import subprocess
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import imageio as imageio
from scipy.io import loadmat

from eval_util import evaluate_hovernet


def mat_to_tiff(path):
    pred_mat_paths = [p for p in natsorted(glob(os.path.join(path, "mat", "*.mat")))]
    for mpath in tqdm(pred_mat_paths, desc="Preprocessing labels"):
        pred_path = os.path.join(path, os.path.basename(mpath.replace(".mat", ".tiff")))
        pred = loadmat(mpath)["inst_map"]
        imageio.imwrite(pred_path, pred)


def run_inference(input_dir, output_dir, dataset, checkpoint_path):
    checkpoint = os.path.basename(checkpoint_path).split("_")[2]
    output_path = os.path.join(output_dir, "inference", dataset, checkpoint)
    input_path = os.path.join(input_dir, dataset, "eval_split", "test_images")

    os.makedirs(output_path, exist_ok=True)
    if checkpoint in ["consep", "cpm17", "kumar"]:
        model_mode = "original"
        nr_types = 0
        type_info = ""
    else:
        model_mode = "fast"
        type_info = os.path.expanduser("~/hover_net/type_info.json")
        if checkpoint == "pannuke":
            nr_types = 6
        else:
            nr_types = 5

    args = [
        "--nr_types", f"{nr_types}",
        "--type_info_path", f"{type_info}",
        "--model_mode", f"{model_mode}",
        "--model_path", f"{checkpoint_path}",
        "--nr_inference_workers", "2",
        "--nr_post_proc_worker", "0",
        "tile",
        "--input_dir", f"{input_path}",
        "--output_dir", f"{output_path}",
        "--save_raw_map",
    ]

    command = ["python3", os.path.expanduser("~/hover_net/run_infer.py")] + args
    print(f"Running inference with HoVerNet {checkpoint} model on {dataset} dataset...")

    subprocess.run(command)
    mat_to_tiff(os.path.join(output_path))
    evaluate_hovernet(
        prediction_dir=output_path,
        label_dir=os.path.join(input_dir, dataset, "eval_split", "test_labels"),
        result_dir=os.path.join(output_dir, "results"),
        checkpoint=checkpoint,
        dataset=dataset,
    )
    shutil.rmtree(os.path.join(output_path, "json"))
    shutil.rmtree(os.path.join(output_path, "mat"))
    shutil.rmtree(os.path.join(output_path, "overlay"))
    print(f"Inference on {dataset} dataset with the HoVerNet {checkpoint} model successfully completed")


def get_hovernet_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None, help="The path to the input data")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="The datasets to infer on")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="The path where the results are saved")
    parser.add_argument(
        "-c", "--checkpoint_path", type=str, default=None,
        help="The path to the HoVerNet checkpoint to use for inference."
    )
    args = parser.parse_args()
    return args


def main():
    args = get_hovernet_args()
    run_inference(
        input_dir=args.input,
        output_dir=args.output_dir,
        dataset=args.dataset,
        checkpoint=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
