import os
import shutil
import argparse
import subprocess


def run_inference(input_dir, output_dir, model_type, dataset, model_name, checkpoint_path, use_masks):
    output_path = os.path.join(output_dir, model_name, "inference", dataset, model_type, "boxes")
    os.makedirs(output_path, exist_ok=True)
    if os.path.exists(
        os.path.join(
            output_dir, model_name, 'results', dataset, 'boxes',
            f'{dataset}_{model_name}_{model_type}_boxes.csv'
        )
    ):
        print(f"Inference with {model_name} model on {dataset} dataset already done")
        return

    input_path = os.path.join(input_dir, dataset, "loaded_testset", "eval_split")
    args = [
        "-m", f"{model_type}",
        "-c", f"{checkpoint_path}",
        "--experiment_folder", f"{output_path}",
        "-i", f"{input_path}",
        "--box",
    ]

    if use_masks:
        args.append("--use_masks")

    command = [
        "python3", os.path.expanduser("~/patho-sam/experiments/patho-sam/evaluate_iterative_prompting.py"),
    ] + args

    print(f"Running inference with {model_name} model (type: {model_type}) on {dataset} dataset...")
    subprocess.run(command)
    shutil.rmtree(os.path.join(output_path, "embeddings"))
    os.makedirs(os.path.join(output_dir, model_name, 'results', dataset, 'boxes'), exist_ok=True)
    shutil.copy(
        os.path.join(
            output_dir, model_name, "inference", dataset, model_type, 'boxes', 'results',
            'iterative_prompting_without_mask', 'iterative_prompts_start_box.csv'
        ),
        os.path.join(
            output_dir, model_name, 'results', dataset, 'boxes', f'{dataset}_{model_name}_{model_type}_boxes.csv'
        )
    )
    print(f"Successfully ran inference with {model_name} model (type: {model_type}) on {dataset} dataset")


def get_iterative_boxes_args():
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
        help="Provide the name of the model to infer with {generalist_sam, pannuke_sam, vanilla_sam, ..}."
    )
    parser.add_argument(
        "-c", "--checkpoint_path", type=str, default=None,
        help="(Optional) provide the path to the checkpoint to use for inference."
    )
    parser.add_argument(
        "--masks_off", action="store_false", help="To disable the usage of logit masks for iterative prompting."
    )


def main():
    args = get_iterative_boxes_args()
    run_inference(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        model_type=args.model,
        dataset=args.dataset,
        model_name=args.name,
        checkpoint_path=args.checkpoint_path,
        use_masks=args.masks_off
    )


if __name__ == "__main__":
    main()
