import os
import shutil
import subprocess
import argparse

from eval_util import evaluate_cellvit


def run_inference(input_dir, output_dir, dataset, checkpoint_path):
    if dataset not in ["pannuke", "nuclick", "srsanet", "lizard", "cpm15", "consep", "cpm17", "monuseg"]:
        data_dir = os.path.join(input_dir, "original_data", dataset, "eval_split")
    else:
        data_dir = os.path.join(input_dir, "vit_data", dataset, "eval_split")

    checkpoint = os.path.splitext(os.path.basename(checkpoint_path))[0].split("-", 1)[1]

    result_dir = os.path.join(output_dir, "results")
    output_path = os.path.join(output_dir, dataset, checkpoint)
    os.makedirs(output_path, exist_ok=True)
    args = [
        "--model", f"{checkpoint_path}",
        "--outdir", f"{output_path}",
        "--magnification", "40",
        "--data", f"{data_dir}",
    ]

    command = [
        "python", os.path.expanduser("~/CellViT/cell_segmentation/inference/inference_cellvit_experiment_monuseg.py"),
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


def get_cellvit_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None, help="The path to the input data")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="The datasets to infer on")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="The path where the results are saved")
    parser.add_argument("-c", "--checkpoint_path", type=str, default=None, help="The checkpoints to use for inference")
    args = parser.parse_args()
    return args


def main():
    args = get_cellvit_args()
    run_inference(
        input_dir=args.input,
        output_dir=args.output_path,
        datasets=[args.dataset],
        checkpoints=[args.checkpoint_path],
    )


if __name__ == "__main__":
    main()
