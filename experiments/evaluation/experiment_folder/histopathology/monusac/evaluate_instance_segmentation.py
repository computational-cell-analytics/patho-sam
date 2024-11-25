import os

from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_instance_segmentation_with_decoder


from util_2 import get_pred_paths, get_default_arguments, VANILLA_MODELS
from evaluate_amg import get_test_paths, get_val_paths


def run_instance_segmentation_with_decoder_inference(model_type, checkpoint, experiment_folder): #removed dataset_name as argument
    val_image_paths, val_gt_paths = get_val_paths()
    test_image_paths, _ = get_test_paths()
from util import get_pred_paths, get_default_arguments, VANILLA_MODELS


def run_instance_segmentation_with_decoder_inference(model_type, checkpoint, experiment_folder, organ_type=None): #removed dataset_name as argument
    val_image_paths, val_gt_paths = get_val_paths(organ_type)
    test_image_paths, _ = get_test_paths(organ_type)

def run_instance_segmentation_with_decoder_inference(model_type, checkpoint, experiment_folder, organ_type=None): #removed dataset_name as argument

from finetuning.specialists.training.histopathology.monusac.util import get_pred_paths, get_default_arguments

def get_test_paths(organ_type=None):
    if organ_type != None:
        image_path = os.path.join('/scratch/users/u11644/data/monusac/loaded_data',f'{organ_type}','images')
        label_path = os.path.join('/scratch/users/u11644/data/monusac/loaded_data',f'{organ_type}','labels')
    else:
        image_path = os.path.join('/scratch/users/u11644/data/monusac/loaded_data/complete_dataset','images')
        label_path = os.path.join('/scratch/users/u11644/data/monusac/loaded_data/complete_dataset','labels')
    image_paths =[]
    gt_paths = []
    for filename in os.listdir(image_path):
        filepath = os.path.join(image_path,filename)
        image_paths.append(filepath)
    image_paths.sort()
    for labelname in os.listdir(label_path):
        gt_path = os.path.join(label_path,labelname)
        gt_paths.append(gt_path)
    gt_paths.sort()
    return image_paths, gt_paths
    
    
def get_val_paths(organ_type=None):
    if organ_type != None:
        image_path = os.path.join('/scratch/users/u11644/data/monusac/loaded_data',f'{organ_type}','val_images')
        label_path = os.path.join('/scratch/users/u11644/data/monusac/loaded_data',f'{organ_type}','val_labels')
    else:
        image_path = os.path.join('/scratch/users/u11644/data/monusac/loaded_data/complete_dataset','val_images')
        label_path = os.path.join('/scratch/users/u11644/data/monusac/loaded_data/complete_dataset','val_labels')
    image_paths =[]
    gt_paths = []
    for filename in os.listdir(image_path):
        filepath = os.path.join(image_path,filename)
        image_paths.append(filepath)
    image_paths.sort()
    for labelname in os.listdir(label_path):
        gt_path = os.path.join(label_path,labelname)
        gt_paths.append(gt_path)
    gt_paths.sort()
    return image_paths, gt_paths


def run_instance_segmentation_with_decoder_inference(model_type, checkpoint, experiment_folder, organ_type=None): #removed dataset_name as argument

    val_image_paths, val_gt_paths = get_val_paths(organ_type)
    test_image_paths, _ = get_test_paths(organ_type)

def run_instance_segmentation_with_decoder_inference(model_type, checkpoint, experiment_folder): #removed dataset_name as argument
    val_image_paths, val_gt_paths = get_val_paths()
    test_image_paths, _ = get_test_paths()

    prediction_folder = run_instance_segmentation_with_decoder(
        checkpoint,
        model_type,
        experiment_folder,
        val_image_paths,
        val_gt_paths,
        test_image_paths
    )
    return prediction_folder


def eval_instance_segmentation_with_decoder(prediction_folder, experiment_folder, dataset): #removed dataset_name as argument
    print("Evaluating", prediction_folder)
    _, gt_paths = get_test_paths(dataset)
    pred_paths = get_pred_paths(prediction_folder)
    save_path = os.path.join(experiment_folder, "results", "instance_segmentation_with_decoder.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)



def main():
    args = get_default_arguments()

    if args.checkpoint is None:
        ckpt = VANILLA_MODELS[args.model]
    else:
        ckpt = args.checkpoint
    prediction_folder = run_instance_segmentation_with_decoder_inference(

        args.model, ckpt, args.experiment_folder
    )
    eval_instance_segmentation_with_decoder(prediction_folder, args.experiment_folder)
if __name__ == "__main__":
    main()
# 46
# 99
# 126
# 153
# 173
# 194
# 202
# 230
# 232
# 284