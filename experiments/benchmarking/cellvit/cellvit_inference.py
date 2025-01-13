import os
import shutil
import subprocess
from eval_util import evaluate_cellvit, zip_predictions, DATASETS


def run_inference(model_dir, input_dir, output_dir, result_dir):
    for dataset in DATASETS:
        data_dir = os.path.join(input_dir, dataset, "loaded_testset")
        for model in ["256-x20", "256-x40", "SAM-H-x20", "SAM-H-x40"]:
            model_path = os.path.join(model_dir, f"CellViT-{model}.pth")
            if os.path.exists(os.path.join(output_dir, dataset, model, dataset, "inference_masks")):
                continue
            output_path = os.path.join(output_dir, dataset, model)
            os.makedirs(output_path, exist_ok=True)
            args = [
                "--model",
                f"{model_path}",
                "--outdir",
                f"{output_path}",
                "--magnification",
                "40",
                "--data",
                f"{data_dir}",
            ]

            command = [
                "python3",
                "/user/titus.griebel/u12649/CellViT/cell_segmentation/inference/inference_cellvit_experiment_monuseg.py",
            ] + args
            print(f"Running inference with CellViT {model} model on {dataset} dataset...")
            subprocess.run(command)
            plot_dir = os.path.join(output_dir, dataset, model, dataset, "plots")
            if os.path.exists(plot_dir):
                shutil.rmtree(plot_dir)
            evaluate_cellvit(output_path, model, dataset, result_dir)
            zip_path = os.path.join(output_dir, dataset)
            zip_predictions(output_path, zip_path)
            shutil.rmtree(output_path)
            print(f"Successfully ran inference with CellViT {model} model on {dataset} dataset")


def main():
    run_inference(
        "/mnt/lustre-grete/usr/u12649/models/cellvit/checkpoints",
        "/mnt/lustre-grete/usr/u12649/data/final_test",
        "/mnt/lustre-grete/usr/u12649/models/cellvit/inference/",
        "/mnt/lustre-grete/usr/u12649/models/cellvit/results",
    )


if __name__ == "__main__":
    main()
