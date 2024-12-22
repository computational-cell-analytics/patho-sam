import os
import shutil
import subprocess

MODEL_NAMES = ["generalist_sam", "pannuke_sam", "vanilla_sam"]


def run_boxes_inference(model_dir, input_dir):
    for model in MODEL_NAMES:
        if model == "vanilla_sam":
            checkpoint_path = None
        else:
            checkpoint_path = os.path.join(model_dir, model, "checkpoints", "best.pt")
        for dataset in [
            "cpm15",
            "cpm17",
            "cryonuseg",
            "janowczyk",
            "lynsec",
            "lizard",
            "monusac",
            "monuseg",
            "nuinsseg",
            "pannuke",
            "puma",
            "tnbc",
        ]:
            output_path = os.path.join(model_dir, model, "inference", dataset, "points")
            os.makedirs(output_path, exist_ok=True)
            if os.path.exists(
                os.path.join(
                    output_path, "results", "iterative_prompting_without_mask", "iterative_prompts_start_point.csv"
                )
            ):
                print(f"Inference with {model} model on {dataset} dataset already done")
                continue
            args = [
                "-m",
                "vit_b",
                "-c",
                f"{checkpoint_path}",
                "-d",
                f"{dataset}",
                "--experiment_folder",
                f"{output_path}",
                "-i",
                f"{input_dir}",
            ]
            command = [
                "python3",
                "/user/titus.griebel/u12649/patho-sam/experiments/histopathology/evaluate_iterative_prompting.py",
            ] + args
            print(f"Running inference with {model} model on {dataset} dataset...")
            subprocess.run(command)
            embedding_path = os.path.join(output_path, "embeddings")
            if os.path.exists(embedding_path):
                shutil.rmtree(embedding_path)

            print(f"Successfully ran iterative points inference with pannuke_sam model on {dataset} dataset")


run_boxes_inference(
    model_dir="/mnt/lustre-grete/usr/u12649/scratch/models",
    input_dir="/mnt/lustre-grete/usr/u12649/scratch/data/test",
)
