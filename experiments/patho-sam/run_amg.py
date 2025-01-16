import os
import shutil
import subprocess
from util import get_inference_args, SAM_TYPES, DATASETS, MODEL_NAMES


def run_inference(model_dir, input_dir, model_types, datasets, model_names):
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
                        f"No checkpoint for {model} model (type: {model_type}) found. Continuing with existent models... "
                    )
                    continue
            for dataset in datasets:
                output_path = os.path.join(model_dir, model, "inference", dataset, model_type, "amg")
                os.makedirs(output_path, exist_ok=True)
                if os.path.exists(os.path.join(model_dir, model, 'results', dataset, 'amg', f'{dataset}_{model}_{model_type}_amg.csv')):
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
                command = [
                    "python3",
                    "/user/titus.griebel/u12649/patho-sam/experiments/patho-sam/evaluate_amg.py",
                ] + args
                print(f"Running inference with {model} model (type: {model_type}) on {dataset} dataset...")
                subprocess.run(command)
                shutil.rmtree(os.path.join(output_path, "embeddings"))
                os.makedirs(os.path.join(model_dir, model, 'results', dataset, 'amg'), exist_ok=True)
                shutil.copy(os.path.join(model_dir, model, "inference", dataset, model_type, 'amg', 'results', 'amg.csv'), 
                            os.path.join(model_dir, model, 'results', dataset, 'amg', f'{dataset}_{model}_{model_type}_amg.csv'))
                print(f"Successfully ran amg inference with {model} model (type: {model_type}) on {dataset} dataset")


def main():
    args = get_inference_args()
    run_inference(
        model_dir="/mnt/lustre-grete/usr/u12649/models",
        input_dir="/mnt/lustre-grete/usr/u12649/data/final_test",
        model_types=[args.model],
        datasets=[args.dataset],
        model_names=[args.name],
    )

if __name__ == "__main__":
    main()
