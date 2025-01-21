import os
import shutil
import subprocess
CVT_CP = [
        "256-x20",
        "256-x40",
        "SAM-H-x20",
        "SAM-H-x40",
    ]

def run_inference(model_dir, input_dir, output_dir, result_dir):
    for dataset in ['pannuke']:
        data_dir = os.path.join(input_dir, dataset, "loaded_testset")
        for checkpoint in CVT_CP:
            model_path = os.path.join(model_dir, f"CellViT-{checkpoint}.pth")
            if os.path.exists(os.path.join(result_dir, dataset, checkpoint, 'ais_result.csv')):
                    print(f"Inference with CellViT model (type: {checkpoint}) on {dataset} dataset already done")
                    continue
            output_path = os.path.join(output_dir, dataset, checkpoint)
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
                "/user/titus.griebel/u12649/CellViT/cell_segmentation/inference/inference_cellvit_experiment_pannuke.py",
            ] + args
            print(f"Running inference with CellViT {checkpoint} model on {dataset} dataset...")
            subprocess.run(command)
            plot_dir = os.path.join(output_path, "plots")
            if os.path.exists(plot_dir):
                shutil.rmtree(plot_dir)
            print(f"Successfully ran inference with CellViT {checkpoint} model on {dataset} dataset")


def main():
    run_inference(
        "/mnt/lustre-grete/usr/u12649/models/cellvit/checkpoints",
        "/mnt/lustre-grete/usr/u12649/data/semantic_data",
        "/mnt/lustre-grete/usr/u12649/models/cellvit_types/inference/",
        "/mnt/lustre-grete/usr/u12649/models/cellvit_types/results",
    )


if __name__ == "__main__":
    main()