import os
import shutil
import subprocess

SAM_SIZES = ["vit_b", "vit_t", "vit_l", "vit_h"]
MODEL_NAMES = ["generalist_sam", "pannuke_sam"]
DATASETS = [
    "cpm15",
    "cpm17",
    "cryonuseg",
    "janowczyk",
    "lynsec_he",
    "lynsec_ihc",
    "lizard",
    "monusac",
    "monuseg",
    "nuinsseg",
    "pannuke",
    "puma",
    "tnbc",
]


def run_inference(model_dir, input_dir):
    for model in MODEL_NAMES:
        checkpoint_path = os.path.join(model_dir, model, "checkpoints", "best.pt")
        for dataset in DATASETS:
            output_path = os.path.join(model_dir, model, "inference", dataset, "instance")
            os.makedirs(output_path, exist_ok=True)
            if os.path.exists(os.path.join(output_path, "results", "instance_segmentation_with_decoder.csv")):
                print(f"Inference with {model} model on {dataset} dataset already done")
                continue
            input_path = os.path.join(input_dir, dataset, "loaded_testset", "eval_split")
            args = [
                "-m",
                "vit_b",
                "-c",
                f"{checkpoint_path}",
                "--experiment_folder",
                f"{output_path}",
                "-i",
                f"{input_path}",
            ]
            command = [
                "python3",
                "/user/titus.griebel/u12649/patho-sam/experiments/histopathology/evaluate_ais.py",
            ] + args
            print(f"Running inference with {model} model on {dataset} dataset...")
            subprocess.run(command)
            embedding_path = os.path.join(output_path, "embeddings")
            if os.path.exists(embedding_path):
                shutil.rmtree(embedding_path)

            print(f"Successfully ran inference with {model} model on {dataset} dataset")


run_inference(
    model_dir="/mnt/lustre-grete/usr/u12649/scratch/models",
    input_dir="/mnt/lustre-grete/usr/u12649/scratch/data/final_test",
)
