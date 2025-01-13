import os

import pandas as pd
from natsort import natsorted


SAM_TYPES = ["vit_b", "vit_l", "vit_h", "vit_b_lm"]

SAM_MODELS = ["generalist_sam", "old_generalist_sam", "lm_sam", "pannuke_sam", "vanilla_sam"]

MODEL_NAMES = ["hovernet", "cellvit", "hovernext", "stardist"] + SAM_MODELS

ALIAS = []

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

CVT_CP = [
        "256-x20",
        "256-x40",
        "SAM-H-x20",
        "SAM-H-x40",
]

HVNT_CP = [
    'consep',
    'cpm17',
    'kumar',
    'monusac',
    'pannuke',
]

CHECKPOINTS = {
    'hovernet':HVNT_CP,
    'hovernext':HNXT_CP,
    'cellvit':CVT_CP, 
    'generalist_sam':SAM_TYPES,
    'pannuke_sam':SAM_TYPES,
    'old_generalist_sam':SAM_TYPES,
    'lm_sam':['vit_b_lm'],
    'stardist':['stardist'],
}

def get_instance_results(path, model_names, overwrite=False):
    for model in model_names:  # iterates over model types
        if model == "vanilla_sam":
            continue
        for checkpoint in CHECKPOINTS[model]:
            result_dict = {"dataset": [], "msa": [], "sa50": [], "sa75": []}
            os.makedirs(os.path.join(path, "sum_results", checkpoint), exist_ok=True)
            csv_out = os.path.join(path, "sum_results", checkpoint, f"ais_{model}_{checkpoint}_results.csv")
            if os.path.exists(csv_out) and not overwrite:
                print(f"{csv_out} already exists.")
                return
            for dataset in natsorted(DATASETS):
                if model in SAM_MODELS:
                    csv_path = os.path.join(
                        path, model, "results", dataset, "instance", f"{dataset}_{model}_{checkpoint}_instance.csv"
                    )
                else:
                    csv_path = os.path.join(path, model, "results", dataset, checkpoint, "ais_result.csv")
                if not os.path.exists(csv_path):
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


def read_interactive(path, model_names=None, overwrite=False):
    for mode in ['boxes', 'points']:
        for model_name in model_names:
            for model_type in SAM_TYPES:
                csv_path = os.path.join(path, "sum_results", model_type, f"{mode}_{model_name}_{model_type}_results.csv")
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
                        path, model_name, "results", dataset, mode, f"{dataset}_{model_name}_{model_type}_{mode}.csv"
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

def concatenate_interactive(model_dir):
    for mode in ['amg', 'points', 'boxes']:
        final_df = pd.DataFrame()
        for sam in SAM_MODELS:
            for model_type in SAM_TYPES:
                try:
                    df = pd.read_csv(os.path.join(model_dir, 'sum_results', model_type, f'{mode}_{sam}_{model_type}_results.csv'))
                except FileNotFoundError:
                    continue
                if mode in ['boxes', 'points']:
                    df = df[['dataset', 'msa_1st', 'msa_8th']]
                    df.rename(columns={'msa_1st':f'{sam}_{model_type}_{mode}', 'msa_8th':f'{sam}_{model_type}_I{mode[0]}'}, inplace=True)
                else:
                    df = df[['dataset', 'msa']]
                    df.rename(columns={'msa':f'{sam}_{model_type}'}, inplace=True)
                if final_df.empty:
                    final_df = df
                else:
                    final_df = pd.merge(final_df, df, on='dataset', how='outer')
        final_df = final_df.dropna(axis=1, how='all')
        final_df.to_csv(os.path.join(model_dir, 'result_test', 'sum_results', f'concatenated_{mode}_results.csv'))
        print(final_df) 


def concatenate_automatic(model_dir): # does not include amg for now
    final_df = pd.DataFrame()
    for model in MODEL_NAMES:
        for checkpoint in CHECKPOINTS[model]:
            try:
                df = pd.read_csv(os.path.join(model_dir, 'sum_results', checkpoint, f'ais_{model}_{checkpoint}_results.csv'))
            except FileNotFoundError:
                print(model, checkpoint)
                continue
            df = df[['dataset', 'msa']]
            df.rename(columns={'msa':f'{model}_{checkpoint}'}, inplace=True)
            if final_df.empty:
                final_df = df
            else:
                final_df = pd.merge(final_df, df, on='dataset', how='outer')
    final_df.to_csv(os.path.join(model_dir, 'result_test', 'sum_results', 'concatenated_ais_results.csv'))

EVAL_PATH = "/mnt/lustre-grete/usr/u12649/models/"


def main():
    # get_instance_results(EVAL_PATH, MODEL_NAMES, overwrite=True)
    # read_amg_csv(EVAL_PATH, SAM_MODELS, overwrite=True)
    read_interactive(EVAL_PATH, SAM_MODELS, overwrite=True)
    # read_it_points_csv(EVAL_PATH, SAM_MODELS, overwrite=False)
    # concatenate_interactive(EVAL_PATH)
    # concatenate_automatic(EVAL_PATH)


if __name__ == "__main__":
    main()
