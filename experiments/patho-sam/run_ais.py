import os
import shutil
import argparse
import subprocess

from util import TILING_WINDOW_DS, PADDING_DS


def run_inference(input_dir, output_dir, model_type, dataset, model_name, checkpoint_path):
    output_path = os.path.join(output_dir, model_name, "inference", dataset, model_type, "instance")
    os.makedirs(output_path, exist_ok=True)
    if dataset not in PADDING_DS:
        input_path = os.path.join(input_dir, "original_data", dataset, "eval_split")
    else:
        input_path = os.path.join(input_dir, "vit_data", dataset, "eval_split")

    args = [
        "-m", f"{model_type}",
        "-c", f"{checkpoint_path}",
        "--experiment_folder", f"{output_path}",
        "-i", f"{input_path}",
    ]

    if dataset in TILING_WINDOW_DS:
        args.append("--tiling_window")
    command = [
        "python3", os.path.expanduser("~/patho-sam/experiments/patho-sam/evaluate_ais.py"),
    ] + args
    print(f"Running inference with {model_name} model (type: {model_type}) on {dataset} dataset...")
    subprocess.run(command)
    os.makedirs(os.path.join(output_dir, model_name, 'results', dataset, 'ais'), exist_ok=True)

    shutil.copy(
        os.path.join(
            output_dir, model_name, "inference", dataset, model_type, 'instance',
            'results', 'instance_segmentation_with_decoder.csv'
        ),
        os.path.join(output_dir, model_name, 'results', dataset, 'ais', f'{dataset}_{model_name}_{model_type}_ais.csv')
    )

    print(f"Successfully ran inference with {model_name} model (type: {model_type}) on {dataset} dataset")


def get_ais_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, default=None, help="The dataset to infer on."
    )
    parser.add_argument(
        "-m", "--model", type=str, default=None, help="Provide the model type to infer with {vit_b, vit_l, vit_h}."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=None, help="Provide path where the results will be stored."
    )
    parser.add_argument(
        "-i", "--input_dir", type=str, default=None, help="Provide path where the dataset is located."
    )
    parser.add_argument(
        "-n", "--name", type=str, default=None,
        help="Provide the name of the model to infer with {generalist_sam, vanilla_sam, ..}."
    )
    parser.add_argument(
        "-c", "--checkpoint_path", type=str, default=None,
        help="(Optional) provide the path to the checkpoint to use for inference."
    )


def main():
    args = get_ais_args()
    run_inference(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        model_type=args.model,
        dataset=args.dataset,
        model_name=args.name,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
