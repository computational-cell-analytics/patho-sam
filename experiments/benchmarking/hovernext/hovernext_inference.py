import os
import subprocess

# This script does not work right now; use the same script inside the hovernext clone


def run_inference(input_dir, output_dir):
    for dataset in [
        "cpm15",
        "cpm17",
        "cryonuseg",
        "janowczyk",
        "lizard",
        "lynsec",
        "monusac",
        "monuseg",
        "nuinsseg",
        "pannuke",
        "puma",
        "tnbc",
    ]:
        for model in ["pannuke_convnextv2_tiny_1", "pannuke_convnextv2_tiny_2", "pannuke_convnextv2_tiny_3"]:
            output_path = os.path.join(output_dir, dataset, model)
            input_path = os.path.join(input_dir, dataset, "loaded_dataset/complete_dataset/eval_split/test_images/*")
            if len(os.listdir(output_path)) > 0:
                continue
            os.makedirs(output_path, exist_ok=True)
            args = [
                "--input", f"{input_path}",
                "--cp", f"{model}",
                "--output_root", f"{output_path}",
                "--tile_size", "512",
            ]

            command = ["python3", "/user/titus.griebel/u12649/hover_next_inference/main.py"] + args
            print(f"Running inference with HoVerNeXt {model} model on {dataset} dataset...")
            subprocess.run(command)
            print(f"Inference on {dataset} dataset with the HoVerNeXt model {model} successfully completed")


run_inference(
    input_dir="/mnt/lustre-grete/usr/u12649/data/test",
    output_dir="/mnt/lustre-grete/usr/u12649/models/hovernext/inference",
)
