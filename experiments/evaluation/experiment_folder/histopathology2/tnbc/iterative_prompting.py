import os

from micro_sam.evaluation import inference
from micro_sam.evaluation.evaluation import run_evaluation_for_iterative_prompting
from evaluate_amg import get_test_paths
#from util import get_paths  # comment this and create a custom function with the same name to run int. seg. on your data
from util_2 import get_model, get_default_arguments


def _run_iterative_prompting(exp_folder, predictor, start_with_box_prompt, use_masks, dataset):
    prediction_root = os.path.join(
        exp_folder, "start_with_box" if start_with_box_prompt else "start_with_point"
    )
    embedding_folder = os.path.join(exp_folder, "embeddings")
    image_paths, gt_paths = get_test_paths(dataset)
    inference.run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=gt_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_root,
        start_with_box_prompt=start_with_box_prompt,
        use_masks=use_masks
    )
    return prediction_root


def _evaluate_iterative_prompting(prediction_root, start_with_box_prompt, exp_folder, dataset):
    _, gt_paths = get_test_paths(dataset)

    run_evaluation_for_iterative_prompting(
        gt_paths=gt_paths,
        prediction_root=prediction_root,
        experiment_folder=exp_folder,
        start_with_box_prompt=start_with_box_prompt,
    )


def main():
    args = get_default_arguments()

    start_with_box_prompt = args.box  # overwrite to start first iters' prompt with box instead of single point

    # get the predictor to perform inference
    predictor = get_model(model_type=args.model, ckpt=args.checkpoint)

    prediction_root = _run_iterative_prompting(
        args.experiment_folder, predictor, start_with_box_prompt, args.use_masks, args.dataset
    )
    _evaluate_iterative_prompting(prediction_root, start_with_box_prompt, args.experiment_folder, args.dataset)


if __name__ == "__main__":
    main()
