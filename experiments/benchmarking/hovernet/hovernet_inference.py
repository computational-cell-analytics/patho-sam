import os
import shutil
import subprocess
import argparse
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import imageio as imageio
from scipy.io import loadmat

from eval_util import evaluate_all_datasets_hovernet, DATASETS


def mat_to_tiff(path):
    pred_mat_paths = [p for p in natsorted(glob(os.path.join(path, "mat", "*.mat")))]
    for mpath in tqdm(pred_mat_paths, desc="Preprocessing labels"):
        pred_path = os.path.join(path, os.path.basename(mpath.replace(".mat", ".tiff")))
        pred = loadmat(mpath)["inst_map"]
        imageio.imwrite(pred_path, pred)


def run_inference(model_dir, input_dir, output_dir, datasets, checkpoint):
    if datasets is None:
        datasets = DATASETS
    for dataset in datasets:
        output_path = os.path.join(output_dir, "inference", dataset, checkpoint)
        input_path = os.path.join(input_dir, dataset, "eval_split", "test_images")
        if os.path.exists(os.path.join(output_dir, "results", dataset, checkpoint,
                                       f'{dataset}_hovernet_{checkpoint}_ais_result.csv')):
            print(f"Inference with HoVerNet model (type: {checkpoint}) on {dataset} dataset already done")
            continue

        os.makedirs(output_path, exist_ok=True)
        if checkpoint in ["consep", "cpm17", "kumar"]:
            model_mode = "original"
            model_path = os.path.join(
                model_dir, f"hovernet_original_{checkpoint}_notype_tf2pytorch.tar"
            )
            nr_types = 0
            type_info = ""
        else:
            model_mode = "fast"

            model_path = os.path.join(model_dir, f"hovernet_fast_{checkpoint}_type_tf2pytorch.tar")
            type_info = os.path.expanduser("~/hover_net/type_info.json")
            if checkpoint == "pannuke":
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

        command = ["python3", os.path.expanduser("~/hover_net/run_infer.py")] + args
        print(f"Running inference with HoVerNet {checkpoint} model on {dataset} dataset...")

        subprocess.run(command)
        mat_to_tiff(os.path.join(output_path))
        evaluate_all_datasets_hovernet(
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
    parser.add_argument("-m", "--model_dir", type=str, default=None, help="The path to the checkpoints")
    parser.add_argument("-i", "--input", type=str, default=None, help="The path to the input data")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="The datasets to infer on")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="The path where the results are saved")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="The HoVerNet checkpoint to use for inference.")
    args = parser.parse_args()
    return args


def main():
    args = get_hovernet_args()
    run_inference(
        model_dir=args.model_dir,
        input_dir=args.input,
        output_dir=args.output_dir,
        dataset=[args.datasets],
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
