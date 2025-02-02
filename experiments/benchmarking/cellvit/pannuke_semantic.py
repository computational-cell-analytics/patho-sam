import os
import shutil
import argparse
import subprocess


CVT_CP = ["256-x20", "256-x40", "SAM-H-x20", "SAM-H-x40"]


def run_inference(checkpoint_path, input_dir, output_dir, dataset):
    data_dir = os.path.join(input_dir, dataset, "eval_split")
    checkpoint = os.path.splitext((os.path.basename(checkpoint_path)[8:]))[0]
    output_path = os.path.join(output_dir, dataset, checkpoint)
    os.makedirs(output_path, exist_ok=True)
    args = [
        "--model", f"{checkpoint_path}",
        "--outdir", f"{output_path}",
        "--magnification", "40",
        "--data", f"{data_dir}",
    ]

    command = [
        "python3",
        os.path.expanduser("~/CellViT/cell_segmentation/inference/inference_cellvit_experiment_pannuke.py"),  # noqa
    ] + args

    print(f"Running inference with CellViT {checkpoint} model on {dataset} dataset...")
    subprocess.run(command)
    plot_dir = os.path.join(output_path, "plots")
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)

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
        datasets=args.dataset,
        checkpoints=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
