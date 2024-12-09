import subprocess
import os
import shutil
def run_inference(model_dir, input_dir, output_dir):
    for dataset in ['cpm15', 'cpm17', 'cryonuseg', 'janowczyk', 'lizard', 'lynsec', 'monusac', 'monuseg', 'nuinsseg', 'pannuke', 'puma', 'tnbc']:
        if os.path.exists(os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/inference/', dataset, model, dataset, 'inference_masks')):
            continue
        output_path = os.path.join(output_dir, dataset, model)
        os.makedirs(output_path, exist_ok=True)
        args = [
            "-m", "vit_b",
            "-d", f"{dataset}",
            "--experiment_folder", f"{output_path}",
            "-i", f"{input_dir}",
        ]
        command = ['python3', '/user/titus.griebel/u12649/patho-sam/experiments/histopathology/evaluate_ais.py'] + args
        print(f'Running inference with CellViT {model} model on {dataset} dataset...')
        subprocess.run(command)
      
        print(f'Successfully ran inference with pannuke_SAM model on {dataset} dataset')


run_inference(model_dir='/mnt/lustre-grete/usr/u12649/scratch/models/pannuke_sam/checkpoints/best.pt',
              input_dir='/mnt/lustre-grete/usr/u12649/scratch/data/test',
              output_dir='/mnt/lustre-grete/usr/u12649/scratch/models/pannuke_sam/inference/')
