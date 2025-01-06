import os
import shutil
import subprocess


SAM_TYPES = ["vit_b", "vit_l", "vit_h"]
MODEL_NAMES = ["generalist_sam", "vanilla_sam"]
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
            if model == "vanilla_sam":
                checkpoint_path = None
            else:
                checkpoint_path = os.path.join(model_dir, model, "checkpoints", model_type, "best.pt")
            for dataset in DATASETS:
                output_path = os.path.join(model_dir, model, "inference", dataset, model_type, "amg")
                os.makedirs(output_path, exist_ok=True)
                if os.path.exists(os.path.join(output_path, "results", "amg.csv")):
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
                    "/user/titus.griebel/u12649/patho-sam/experiments/patho-sam/evaluation/evaluate_amg.py",
                ] + args
                print(f"Running inference with {model} model (type: {model_type}) on {dataset} dataset...")
                subprocess.run(command)
                embedding_path = os.path.join(output_path, "embeddings")
                if os.path.exists(embedding_path):
                    shutil.rmtree(embedding_path)

                print(f"Successfully ran amg inference with {model} model (type: {model_type}) on {dataset} dataset")


run_boxes_inference(
    model_dir="/mnt/lustre-grete/usr/u12649/models",
    input_dir="/mnt/lustre-grete/usr/u12649/data/final_test",
)
