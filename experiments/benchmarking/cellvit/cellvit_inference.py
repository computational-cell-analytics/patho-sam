import os
import shutil
import subprocess
import argparse

from eval_util import evaluate_cellvit, DATASETS


CHECKPOINTS = ["256-x20", "256-x40", "SAM-H-x20", "SAM-H-x40"]


def run_inference(model_dir, input_dir, output_dir, result_dir, datasets=None, checkpoints=None):
    if datasets is None:
        datasets = DATASETS
    if checkpoints is None:
        checkpoints = CHECKPOINTS
    for dataset in DATASETS:
        if dataset not in ["pannuke", "nuclick", "srsanet", "lizard", "cpm15", "consep", "cpm17", "monuseg"]:
            data_dir = os.path.join(input_dir, "original_data", dataset, "eval_split")
        else:
            data_dir = os.path.join(input_dir, "vit_data", dataset, "eval_split")

        for checkpoint in checkpoints:  # ["256-x20", "256-x40", "SAM-H-x20", "SAM-H-x40"]:
            model_path = os.path.join(model_dir, f"CellViT-{checkpoint}.pth")
            if os.path.exists(
                os.path.join(result_dir, dataset, checkpoint, f'{dataset}_cellvit_{checkpoint}_ais_result.csv')
            ):
                print(f"Inference with CellViT model (type: {checkpoint}) on {dataset} dataset already done")
                continue

            output_path = os.path.join(output_dir, dataset, checkpoint)
            os.makedirs(output_path, exist_ok=True)
            args = [
                "--model", f"{model_path}",
                "--outdir", f"{output_path}",
                "--magnification", "40",
                "--data", f"{data_dir}",
            ]
            if dataset == "lizard":
                args.append("--stitched")

            command = [
                "python",
                os.path.expanduser("~/CellViT/cell_segmentation/inference/inference_cellvit_experiment_monuseg.py"),
            ] + args

            print(f"Running inference with CellViT {checkpoint} model on {dataset} dataset...")
            subprocess.run(command)

            plot_dir = os.path.join(output_dir, dataset, checkpoint, dataset, "plots")
            if os.path.exists(plot_dir):
                shutil.rmtree(plot_dir)

            if os.path.exists(os.path.join(output_path, "inference_monuseg.log")):
                os.remove(os.path.join(output_path, "inference_monuseg.log"))

            evaluate_cellvit(output_path, checkpoint, dataset, data_dir, result_dir)
            print(f"Successfully ran inference with CellViT {checkpoint} model on {dataset} dataset")


def main():
    run_inference(
        "/mnt/lustre-grete/usr/u12649/models/cellvit/checkpoints",
        "/mnt/lustre-grete/usr/u12649/data/",
        "/mnt/lustre-grete/usr/u12649/models/cellvit/inference/",
        "/mnt/lustre-grete/usr/u12649/models/cellvit/results",
    )


# def get_cellvit_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-m", "--model_dir", type=str, default=None, help="The path to the checkpoints")
#     parser.add_argument("-i", "--input", type=str, default=None, help="The path to the input data")
#     parser.add_argument("-d", "--dataset", type=str, default=None, help="The datasets to infer on")
#     parser.add_argument("-o", "--output_path", type=str, default=None, help="The path where the results are saved")
#     parser.add_argument("-c", "--checkpoint", type=str, default=None, help="The checkpoints to use for inference")
#     args = parser.parse_args()
#     return args


# def main():
#     args = get_cellvit_args()
#     run_inference(
#         model_dir=args.model_dir,
#         input_dir=args.input,
#         output_dir=args.output_path,
#         datasets=[args.dataset],
#         checkpoints=[args.checkpoint],
#     )


if __name__ == "__main__":
    main()
