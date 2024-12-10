import os
from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_amg
from util import get_default_arguments, get_pred_paths, VANILLA_MODELS, get_test_paths, get_val_paths


def run_amg_inference(model_type, checkpoint, experiment_folder, dataset, input_path):
    val_image_paths, val_gt_paths = get_val_paths(input_path, dataset)
    test_image_paths, _ = get_test_paths(input_path, dataset)
    prediction_folder = run_amg(
        checkpoint,
        model_type,
        experiment_folder,
        val_image_paths,
        val_gt_paths,
        test_image_paths
    )
    return prediction_folder


def eval_amg(prediction_folder, experiment_folder, dataset, input_path):
    print("Evaluating", prediction_folder)
    _, gt_paths = get_test_paths(input_path, dataset)
    pred_paths = get_pred_paths(prediction_folder)
    save_path = os.path.join(experiment_folder, "results", "amg.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    args = get_default_arguments()
    if args.checkpoint is None:
        ckpt = VANILLA_MODELS[args.model]
    else:
        ckpt = args.checkpoint

    prediction_folder = run_amg_inference(args.model, ckpt, args.experiment_folder, args.dataset, args.input_path)
    eval_amg(prediction_folder, args.experiment_folder, args.dataset, args.input_path)


if __name__ == "__main__":
    main()
