import os
import shutil
import subprocess

from util import get_inference_args, SAM_TYPES, DATASETS, MODEL_NAMES


def run_inference(model_dir, input_dir, model_types, datasets, model_names, use_masks=False):
    if model_types == [None]:
        model_types = SAM_TYPES
    if datasets == [None]:
        datasets = DATASETS
    if model_names == [None]:
        model_names = MODEL_NAMES
    for model in model_names:
        for model_type in model_types:
            if model in ["vanilla_sam", "lm_sam"]:
                checkpoint_path = None
                if model == "lm_sam":
                    if model_type != "vit_b":
                        continue
                    model_type = "vit_b_lm"
            else:
                checkpoint_path = os.path.join(model_dir, model, "checkpoints", model_type, "best.pt")
                if not os.path.exists(checkpoint_path):
                    print(
                        f"No checkpoint for {model} model (type: {model_type}) found. "
                        "Continuing with existent models... "
                    )
                    continue
            for dataset in datasets:
                output_path = os.path.join(model_dir, model, "inference", dataset, model_type, "points")
                os.makedirs(output_path, exist_ok=True)
                if os.path.exists(
                    os.path.join(
                        model_dir, model, 'results', dataset, 'points',
                        f'{dataset}_{model}_{model_type}_points.csv'
                    )
                ):
                    print(f"Inference with {model} model (type: {model_type}) on {dataset} dataset already done")
                    continue

                input_path = os.path.join(input_dir, dataset, "loaded_testset", "eval_split")
                args = [
                    "-m",
                    f"{model_type}",
                    "-c",
                    f"{checkpoint_path}",
                    "--experiment_folder",
                    f"{output_path}",
                    "-i",
                    f"{input_path}",
                ]
                if use_masks:
                    args.append("--use_masks")
                command = [
                    "python3",
                    "/user/titus.griebel/u12649/patho-sam/experiments/patho-sam/evaluate_iterative_prompting.py",
                ] + args
                print(f"Running inference with {model} model (type: {model_type}) on {dataset} dataset...")
                subprocess.run(command)
                embedding_path = os.path.join(output_path, "embeddings")
                shutil.rmtree(embedding_path)
                os.makedirs(os.path.join(model_dir, model, 'results', dataset, 'points'), exist_ok=True)
                shutil.copy(
                    os.path.join(
                        model_dir, model, "inference", dataset, model_type, 'points', 'results',
                        'iterative_prompting_without_mask', 'iterative_prompts_start_point.csv'
                    ),
                    os.path.join(
                        model_dir, model, 'results', dataset, 'points',
                        f'{dataset}_{model}_{model_type}_points.csv'
                    )
                )
                print(
                    f"Successfully ran iterative points inference with {model} "
                    "model (type: {model_type}) on {dataset} dataset"
                )


def main():
    args = get_inference_args()
    run_inference(
        model_dir="/mnt/lustre-grete/usr/u12649/models",
        input_dir="/mnt/lustre-grete/usr/u12649/data/final_test",
        model_types=[args.model],
        datasets=[args.dataset],
        model_names=[args.name],
        use_masks=args.use_masks,
    )


if __name__ == "__main__":
    main()
