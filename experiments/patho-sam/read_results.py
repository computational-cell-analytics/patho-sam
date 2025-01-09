import os

import pandas as pd
from natsort import natsorted


SAM_TYPES = ["vit_b", "vit_l", "vit_h", "vit_b_lm"]

MODEL_NAMES = ["generalist_sam", "lm_sam", "pannuke_sam", "vanilla_sam", "hovernet", "cellvit", "hovernext"]
EVAL_PATH = "/mnt/lustre-grete/usr/u12649/models/"
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

HNXT_CP = [
    "lizard_convnextv2_large",
    "lizard_convnextv2_base",
    "lizard_convnextv2_tiny",
    "pannuke_convnextv2_tiny_1",
    "pannuke_convnextv2_tiny_2",
    "pannuke_convnextv2_tiny_3",
]


def get_instance_results(path, model, checkpoint=None, overwrite=False):
    result_dict = {"dataset": [], "msa": [], "sa50": [], "sa75": []}
    os.makedirs(os.path.join(path, "sum_results", checkpoint), exist_ok=True)
    csv_out = os.path.join(path, "sum_results", checkpoint, f"ais_{model}_{checkpoint}_results.csv")
    if os.path.exists(csv_out) and not overwrite:
        print(f"{csv_out} already exists.")
        return
    for dataset in natsorted(DATASETS):
        if model in ["generalist_sam", "pannuke_sam", "lm_sam"]:
            csv_path = os.path.join(
                path, model, "results", dataset, "instance", f"{dataset}_{model}_{checkpoint}_decoder.csv"
            )
        else:
            csv_path = os.path.join(path, model, "results", dataset, checkpoint, "ais_result.csv")
        if not os.path.exists(csv_path):
            if not model == "lm_sam":
                print(
                    f"Ais results for {model} model on {dataset} dataset with checkpoint {checkpoint} not in {csv_path}"
                )
            continue
        df = pd.read_csv(csv_path)
        result_dict["msa"].append(df.loc[0, "mSA"])
        result_dict["sa50"].append(df.loc[0, "SA50"])
        result_dict["sa75"].append(df.loc[0, "SA75"])
        result_dict["dataset"].append(dataset)
    df = pd.DataFrame(result_dict)
    print(
        f"Results of instance segmentation evaluation with {model} model using checkpoint {checkpoint}:\n",
        df.head(len(DATASETS)),
    )
    df.to_csv(csv_out, index=False)


def read_instance_csv(path, model_names, overwrite=False):
    for model in model_names:  # iterates over model types
        if model == "vanilla_sam":
            continue
        elif model in MODEL_NAMES[:3]:
            for model_type in SAM_TYPES:
                get_instance_results(path, model, model_type, overwrite)
        elif model == "cellvit":
            for checkpoint in [
                "256-x20",
                "256-x40",
                "SAM-H-x20",
                "SAM-H-x40",
            ]:  # iterates over specific cellvit checkpoints
                get_instance_results(path, model, checkpoint, overwrite)
        elif model == "hovernet":
            for checkpoint in ["consep", "cpm17", "kumar", "pannuke", "monusac"]:  # Hovernet checkpoints
                get_instance_results(path, model, checkpoint, overwrite)
        elif model == "hovernext":
            for checkpoint in HNXT_CP:
                get_instance_results(path, model, checkpoint, overwrite)


def read_amg_csv(path, model_names, overwrite=False):
    for model_name in model_names:
        for model_type in SAM_TYPES:
            csv_path = os.path.join(path, "sum_results", model_type, f"amg_{model_name}_{model_type}_results.csv")
            if os.path.exists(csv_path) and not overwrite:
                print(f"{csv_path} already exists.")
                continue
            result_dict = {"dataset": [], "msa": [], "sa50": [], "sa75": []}
            for dataset in natsorted(DATASETS):
                dataset_csv = os.path.join(
                    path, model_name, "results", dataset, "amg", f"{dataset}_{model_name}_{model_type}_amg.csv"
                )
                if not os.path.exists(dataset_csv):
                    continue

                df = pd.read_csv(dataset_csv)
                result_dict["msa"].append(df.loc[0, "mSA"])
                result_dict["sa50"].append(df.loc[0, "SA50"])
                result_dict["sa75"].append(df.loc[0, "SA75"])
                result_dict["dataset"].append(dataset)

            df = pd.DataFrame(result_dict)
            print(f"Results of amg evaluation with {model_name} (model_type: {model_type}):\n", df.head(len(DATASETS)))
            os.makedirs(os.path.join(path, model_name, "sum_results"), exist_ok=True)
            df.to_csv(csv_path, index=False)


def read_it_boxes_csv(path, model_names=None, overwrite=False):
    for model_name in model_names:
        for model_type in SAM_TYPES:
            csv_path = os.path.join(path, "sum_results", model_type, f"boxes_{model_name}_{model_type}_results.csv")
            if os.path.exists(csv_path) and not overwrite:
                print(f"{csv_path} already exists.")
                continue
            result_dict = {
                "dataset": [],
                "msa_1st": [],
                "msa_8th": [],
                "sa50_1st": [],
                "sa50_8th": [],
                "sa75_1st": [],
                "sa75_8th": [],
            }
            for dataset in natsorted(DATASETS):
                dataset_csv = os.path.join(
                    path, model_name, "results", dataset, "boxes", f"{dataset}_{model_name}_{model_type}_box.csv"
                )
                if not os.path.exists(dataset_csv):
                    continue

                df = pd.read_csv(dataset_csv)
                result_dict["msa_1st"].append(df.loc[0, "mSA"])
                result_dict["sa50_1st"].append(df.loc[0, "SA50"])
                result_dict["sa75_1st"].append(df.loc[0, "SA75"])
                result_dict["msa_8th"].append(df.loc[7, "mSA"])
                result_dict["sa50_8th"].append(df.loc[7, "SA50"])
                result_dict["sa75_8th"].append(df.loc[7, "SA75"])
                result_dict["dataset"].append(dataset)
            df = pd.DataFrame(result_dict)
            print(f"Results of iterative prompting with boxes evaluation with {model_name} (model_type: {model_type}):")
            print(df.head(len(DATASETS)))
            os.makedirs(os.path.join(path, model_name, "sum_results"), exist_ok=True)
            df.to_csv(csv_path, index=False)


def read_it_points_csv(path, model_names=None, overwrite=False):
    for model_name in model_names:
        for model_type in SAM_TYPES:
            csv_path = os.path.join(path, "sum_results", model_type, f"points_{model_name}_{model_type}_results.csv")
            if os.path.exists(csv_path) and not overwrite:
                print(f"{csv_path} already exists.")
                continue
            result_dict = {
                "dataset": [],
                "msa_1st": [],
                "msa_8th": [],
                "sa50_1st": [],
                "sa50_8th": [],
                "sa75_1st": [],
                "sa75_8th": [],
            }
            for dataset in natsorted(DATASETS):
                dataset_csv = os.path.join(
                    path, model_name, "results", dataset, "points", f"{dataset}_{model_name}_{model_type}_point.csv"
                )

                if not os.path.exists(dataset_csv):
                    print("dataset csv not found")
                    continue
                df = pd.read_csv(dataset_csv)
                result_dict["msa_1st"].append(df.loc[0, "mSA"])
                result_dict["sa50_1st"].append(df.loc[0, "SA50"])
                result_dict["sa75_1st"].append(df.loc[0, "SA75"])
                result_dict["msa_8th"].append(df.loc[7, "mSA"])
                result_dict["sa50_8th"].append(df.loc[7, "SA50"])
                result_dict["sa75_8th"].append(df.loc[7, "SA75"])
                result_dict["dataset"].append(dataset)

            df = pd.DataFrame(result_dict)
            print(
                f"Results of iterative prompting with points evaluation with {model_name} (model_type: {model_type}):"
            )
            print(df.head(len(DATASETS)))
            os.makedirs(os.path.join(path, model_name, "sum_results"), exist_ok=True)
            df.to_csv(csv_path, index=False)


def main():
    # read_instance_csv(EVAL_PATH, MODEL_NAMES, overwrite=True)
    # read_amg_csv(EVAL_PATH, MODEL_NAMES[:4], overwrite=True)
    # read_it_boxes_csv(EVAL_PATH, MODEL_NAMES[:4], overwrite=True)
    read_it_points_csv(EVAL_PATH, MODEL_NAMES[:4], overwrite=True)


if __name__ == "__main__":
    main()
