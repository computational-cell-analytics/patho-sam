import os
import shutil
import subprocess


SAM_TYPES = ["vit_b", "vit_l", "vit_h"]
MODEL_NAMES = ["lm_sam", "old_generalist_sam", "vanilla_sam"]
DATASETS = [
    "consep",
    "cpm15",
    "cpm17",
    "cryonuseg",
    "lizard",
    "lynsec_he",
    "lynsec_ihc",
    "monusac",
    "monuseg",
    "nuclick",
    "nuinsseg",
    "pannuke",
    "puma",
    "srsanet",
    "tnbc",
]


def run_boxes_inference(model_dir, input_dir):
    for model in MODEL_NAMES:
        for model_type in SAM_TYPES:
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
                        f"No checkpoint for {model} model (type: {model_type} found. Continuing with existent models... "
                    )
                    continue
            for dataset in DATASETS:
                output_path = os.path.join(model_dir, model, "inference", dataset, model_type, "boxes")
                os.makedirs(output_path, exist_ok=True)
                if os.path.exists(
                    os.path.join(
                        output_path, "results", "iterative_prompting_without_mask", "iterative_prompts_start_box.csv"
                    )
                ):
                    print(f"Inference with {model} model on {dataset} dataset already done")
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
                    "--box",
                ]
                command = [
                    "python3",
                    "/user/titus.griebel/u12649/patho-sam/experiments/patho-sam/evaluate_iterative_prompting.py",
                ] + args
                print(f"Running inference with {model} model (type: {model_type}) on {dataset} dataset...")
                subprocess.run(command)
                shutil.rmtree(os.path.join(output_path, "embeddings"))
                os.makedirs(os.path.join(model_dir, model, 'results', dataset, 'boxes'), exist_ok=True)
                shutil.copy(os.path.join(model_dir, model, "inference", dataset, model_type, 'boxes', 'results', 'iterative_prompting_without_mask', 'iterative_prompts_start_box.csv'), 
                            os.path.join(model_dir, model, 'results', dataset, 'boxes', f'{dataset}_{model}_{model_type}_boxes.csv'))
                print(f"Successfully ran inference with {model} model (type: {model_type}) on {dataset} dataset")


run_boxes_inference(
    model_dir="/mnt/lustre-grete/usr/u12649/models",
    input_dir="/mnt/lustre-grete/usr/u12649/data/final_test",
)
