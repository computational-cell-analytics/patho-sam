import subprocess
import os
import shutil


DATASETS = [
    'cpm15', 'cpm17', 'cryonuseg', 'janowczyk', 'lizard', 'lynsec',
    'monusac', 'monuseg', 'nuinsseg', 'pannuke', 'puma', 'tnbc'
]


def run_inference(model_dir, input_dir, output_dir):
    for dataset in DATASETS:
        if dataset == 'lizard':
            magnification = 20
        else:
            magnification = 40
        for model in ['256-x20', '256-x40', 'SAM-H-x20', 'SAM-H-x40']:
            model_path = os.path.join(model_dir, f'CellViT-{model}.pth')
            if os.path.exists(os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/inference/', dataset, model, dataset, 'inference_masks')):
                continue
            output_path = os.path.join(output_dir, dataset, model)
            os.makedirs(output_path, exist_ok=True)
            args = [
                "--model", f"{model_path}",
                "--dataset", f"{dataset}",
                "--outdir", f"{output_path}",
                "--magnification", f"{magnification}",
                "--data", f"{input_dir}",
            ]

            command = ['python3', '/user/titus.griebel/u12649/CellViT/cell_segmentation/inference/inference_cellvit_experiment_monuseg.py'].extend(args)
            print(f'Running inference with CellViT {model} model on {dataset} dataset...')
            subprocess.run(command)
            plot_dir = os.path.join(output_dir, dataset, model, dataset, 'plots')
            if os.path.exists(plot_dir):
                shutil.rmtree(plot_dir)
            print(f'Successfully ran inference with CellViT {model} model on {dataset} dataset')


run_inference('/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/checkpoints', '/mnt/lustre-grete/usr/u12649/scratch/data', '/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/inference/')



# def main():
#     for dataset in DATASETS:
#         if os.path.exists(os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/inference/', f'{dataset}')):
#             continue
#         args = [
#             "--model", "/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/checkpoints/CellViT-256-x40.pth",
#             "--dataset", f"{dataset}",
#             "--outdir", "/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/inference",
#             "--magnification", "40",
#             "--data", "/mnt/lustre-grete/usr/u12649/scratch/data",
#         ]

#         command = [
#             'python3',
#             '/user/titus.griebel/u12649/CellViT/cell_segmentation/inference/inference_cellvit_experiment_monuseg.py'
#         ].extend(args)

#         subprocess.run(command)


# if __name__ == "__main__":
#     main()
