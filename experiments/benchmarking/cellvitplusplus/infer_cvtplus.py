import os
import subprocess
import pandas as pd
from glob import glob
from natsort import natsorted

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

CVTPP_CP = [
    # 'Virchow-x40-AMP',
    'SAM-H-x40-AMP',
    # '256-x40-AMP'
]


def run_inference(model_dir, input_dir, output_dir, result_dir, label_dir):
    for dataset in DATASETS:
        data_dir = os.path.join(input_dir, dataset)
        files = {
            "path": list(natsorted(glob(os.path.join(data_dir, '*.tiff')))),
            "slide_mpp": [0.25 for i in range(len(list(natsorted(glob(os.path.join(data_dir, '*.tiff'))))))],
            "magnification": [40 for i in range(len(list(natsorted(glob(os.path.join(data_dir, '*.tiff'))))))]
        }
        filelist_df = pd.DataFrame(files)
        os.makedirs(os.path.join(input_dir, "file_lists"), exist_ok=True)
        csv_filelist = os.path.join(input_dir, "file_lists", f"{dataset}_filelist.csv")
        filelist_df.to_csv(csv_filelist, index=False)

        for checkpoint in CVTPP_CP:
            checkpoint_path = os.path.join(model_dir, f"CellViT-{checkpoint}.pth")
            output_path = os.path.join(output_dir, dataset, checkpoint)
            if os.path.exists(output_path):
                if len(os.listdir(output_path)) > 1:
                    continue
            os.makedirs(output_path, exist_ok=True)
            args = [
                "--model", f"{checkpoint_path}",
                "--outdir", f"{output_path}",
                "process_dataset",
                "--filelist", f"{csv_filelist}"
            ]
            command = [
                "python3",
                "/user/titus.griebel/u12649/CellViT-plus-plus/cellvit/detect_cells.py",
            ] + args
            print(f"Running inference with CellViT-plus-plus {checkpoint} model on {dataset} dataset...")
            subprocess.run(command)

            for file in glob(os.path.join(output_path, '*json')):
                os.remove(file)
            # evaluate_cellvit(output_path, checkpoint, dataset, result_dir, label_dir)

            print(f"Successfully ran inference with CellViT {checkpoint} model on {dataset} dataset")


def main():
    run_inference(
        "/mnt/lustre-grete/usr/u12649/models/cellvit_plusplus/checkpoints",
        "/mnt/lustre-grete/usr/u12649/data/cvtplus/preprocessed",
        "/mnt/lustre-grete/usr/u12649/models/cellvit_plusplus/inference/",
        "/mnt/lustre-grete/usr/u12649/models/cellvit_plusplus/results",
        "/mnt/lustre-grete/usr/u12649/data/final_test"
    )


if __name__ == "__main__":
    main()
