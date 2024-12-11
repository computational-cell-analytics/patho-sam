import subprocess
import os
import shutil
import zipfile

DATASETS = [
    'cpm15', 'cpm17', 'cryonuseg', 'janowczyk', 'lizard', 'lynsec',
    'monusac', 'monuseg', 'nuinsseg', 'pannuke', 'puma', 'tnbc'
]


def zip_results(path, target_dir):
    print(f'Zipping {path}...')
    zip_name = os.path.basename(path) + '.zip'
    zip_path = os.path.join(target_dir, zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=path)
                zipf.write(file_path, arcname)
    print('Successfully zipped results')


def run_inference(model_dir, input_dir, output_dir):
    for dataset in DATASETS:
        if dataset == 'lizard':
            magnification = 20
        else:
            magnification = 40
        for model in ['256-x20', '256-x40', 'SAM-H-x20', 'SAM-H-x40']:
            model_path = os.path.join(model_dir, f'CellViT-{model}.pth')
            if os.path.exists(os.path.join(output_dir, dataset, model, dataset, 'inference_masks')):
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

            command = ['python3', '/user/titus.griebel/u12649/CellViT/cell_segmentation/inference/inference_cellvit_experiment_monuseg.py'] + args
            print(f'Running inference with CellViT {model} model on {dataset} dataset...')
            subprocess.run(command)
            plot_dir = os.path.join(output_dir, dataset, model, dataset, 'plots')
            if os.path.exists(plot_dir):
                shutil.rmtree(plot_dir)
            zip_path = os.path.join(output_dir, dataset)
            zip_results(output_path, zip_path)
            shutil.rmtree(output_path)
            print(f'Successfully ran inference with CellViT {model} model on {dataset} dataset')


run_inference('/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/checkpoints', '/mnt/lustre-grete/usr/u12649/scratch/data', '/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/inference/')



# def main():
#     for dataset in DATASETS:
#         if os.path.exists(os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/inference/', f'{dataset}')):
#             continue   CellViT 256-x20 model on cpm15 dataset
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
