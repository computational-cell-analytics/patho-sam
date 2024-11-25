from cryonuseg import get_cryonuseg_paths
import numpy as np
import os
import imageio
from tqdm import tqdm
import h5py
import os
import argparse
import warnings
from glob import glob

from torch_em.data import datasets

from micro_sam.evaluation.livecell import _get_livecell_paths
import torch
# def test():
#     image_paths, _ = get_cryonuseg_paths('/mnt/lustre-grete/usr/u12649/scratch/data/cryonuseg')
#     max_value = []
#     for image in tqdm(image_paths):
#         image = imageio.imread(image)
#         # image = image.astype(np.float32)
#         print(f'Image datatype: {image.dtype}')
#         unique_values = np.unique(image)
#         # print("Max value:", max(unique_values))
#         # print("Min value:", min(unique_values))
#         max_value.append(max(unique_values))
#     print(f'{max(max_value)} was the maximum value of the given images')


# test()

# def load_checkpoint(cp1, cp2):
#     checkpoint1 = torch.load(cp1, map_location=torch.device('cpu'))
#     checkpoint2 = torch.load(cp2, map_location=torch.device('cpu'))
#     print('Checkpoint vanilla SAM keys:', checkpoint1.keys())
#     print('Checkpoint finetuned SAM keys:', checkpoint2.keys())


# load_checkpoint('/scratch-grete/projects/nim00007/sam/models/vanilla/sam_vit_b_01ec64.pth', '/mnt/lustre-grete/usr/u12649/scratch/models/checkpoints/vit_b/pannuke_sam/best.pt')

# def get_default_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-m", "--model", type=str, required=True, help="Provide the model type to initialize the predictor"
#     )
#     parser.add_argument("-c", "--checkpoint", type=none_or_str, default=None)
#     parser.add_argument("-e", "--experiment_folder", type=str, required=True)
#     parser.add_argument("-d", "--dataset", type=str, default=None)
#     parser.add_argument("--box", action="store_true", help="If passed, starts with first prompt as box")
#     parser.add_argument(
#         "--use_masks", action="store_true", help="To use logits masks for iterative prompting."
#     )
#     parser.add_argument("--peft_rank", default=None, type=int, help="The rank for peft method.")
#     parser.add_argument("--peft_module", default=None, type=str, help="The module for peft method. (e.g. LoRA or FacT)")
#     args = parser.parse_args()
#     return args

from util_3 import get_default_arguments, get_pred_paths
import os
from evaluate_amg_cryonuseg import get_val_paths, get_test_paths
from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_instance_segmentation_with_decoder




def run_instance_segmentation_with_decoder_inference(
    model_type, checkpoint, experiment_folder, peft_kwargs,
):
    val_image_paths, val_gt_paths = get_val_paths()
    test_image_paths, _ = get_test_paths()
    prediction_folder = run_instance_segmentation_with_decoder(
        checkpoint=checkpoint,
        model_type=model_type,
        experiment_folder=experiment_folder,
        val_image_paths=val_image_paths,
        val_gt_paths=val_gt_paths,
        test_image_paths=test_image_paths,
        peft_kwargs=peft_kwargs,
    )
    return prediction_folder


def eval_instance_segmentation_with_decoder(prediction_folder, experiment_folder):
    print("Evaluating", prediction_folder)
    _, gt_paths = get_test_paths()
    pred_paths = get_pred_paths(prediction_folder)
    save_path = os.path.join(experiment_folder, "results", "instance_segmentation_with_decoder.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    args = get_default_arguments()
    peft_kwargs = {"rank": args.peft_rank, "module": args.peft_module}
    prediction_folder = run_instance_segmentation_with_decoder_inference(
        args.model, args.checkpoint, args.experiment_folder, peft_kwargs 
    )
    eval_instance_segmentation_with_decoder(prediction_folder, args.experiment_folder)


if __name__ == "__main__":
    main()
