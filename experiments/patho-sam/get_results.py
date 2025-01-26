import os
from natsort import natsorted

import pandas as pd


SAM_TYPES = ["vit_b", "vit_l", "vit_h", "vit_b_lm"]
SAM_MODELS = ["generalist_sam", "lm_sam", "pannuke_sam", "vanilla_sam", "nuclick_sam", "glas_sam", "cryonuseg_sam"]
MODEL_NAMES = ["hovernet", "cellvit", "hovernext", "stardist", "cellvitpp", "instanseg"] + SAM_MODELS

DATASETS = [
    "consep",
    "cpm15",
    "cpm17",
    "cryonuseg",
    "glas",
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
    "consep",
    "cpm17",
    "kumar",
    "monusac",
    "pannuke",
]

CVTPP_CP = ["SAM-H-x40-AMP"]

CHECKPOINTS = {
    "hovernet": HVNT_CP,
    "hovernext": HNXT_CP,
    "cellvit": CVT_CP,
    "cellvitpp": CVTPP_CP,
    "generalist_sam": SAM_TYPES,
    "pannuke_sam": SAM_TYPES,
    "old_generalist_sam": SAM_TYPES,
    "nuclick_sam": SAM_TYPES,
    "vanilla_sam": SAM_TYPES,
    "glas_sam": SAM_TYPES,
    "cryonuseg_sam": SAM_TYPES,
    "lm_sam": ["vit_b_lm"],
    "stardist": ["stardist"],
    "instanseg": ["instanseg"],
}


def get_results(path, overwrite=False):
    os.makedirs(os.path.join(path, "sum_results", "concatenated_results"), exist_ok=True)
    for mode in ['ais', 'amg', 'boxes', 'points']:
        csv_concat = os.path.join(path, "sum_results", "concatenated_results", f"concatenated_{mode}_results.csv")
        concat_df = pd.DataFrame()
        for model in MODEL_NAMES:
            if model not in SAM_MODELS and mode != 'ais':
                continue

            if model == "vanilla_sam" and mode == 'ais':
                continue

            for checkpoint in CHECKPOINTS[model]:
                if mode in ['ais', 'amg']:
                    result_dict = {"dataset": [], "msa": [], "sa50": [], "sa75": []}
                else:
                    result_dict = {
                        "dataset": [],
                        "msa_1st": [],
                        "msa_8th": [],
                        "sa50_1st": [],
                        "sa50_8th": [],
                        "sa75_1st": [],
                        "sa75_8th": [],
                    }

                os.makedirs(os.path.join(path, "sum_results", model), exist_ok=True)
                csv_out = os.path.join(path, "sum_results", model, f"{mode}_{model}_{checkpoint}_results.csv")
                if os.path.exists(csv_out) and not overwrite:
                    print(f"{csv_out} already exists.")
                    continue

                for dataset in natsorted(DATASETS):
                    if model in SAM_MODELS:
                        csv_path = os.path.join(
                            path, model, "results", dataset, mode, f"{dataset}_{model}_{checkpoint}_{mode}.csv"
                        )
                    else:
                        csv_path = os.path.join(
                            path, model, "results", dataset, checkpoint,
                            f"{dataset}_{model}_{checkpoint}_{mode}_result.csv"
                        )

                    if not os.path.exists(csv_path):
                        continue

                    df = pd.read_csv(csv_path)
                    if mode in ['ais', 'amg']:
                        result_dict["msa"].append(df.loc[0, "mSA"])
                        result_dict["sa50"].append(df.loc[0, "SA50"])
                        result_dict["sa75"].append(df.loc[0, "SA75"])
                        result_dict["dataset"].append(dataset)
                    else:
                        result_dict["msa_1st"].append(df.loc[0, "mSA"])
                        result_dict["sa50_1st"].append(df.loc[0, "SA50"])
                        result_dict["sa75_1st"].append(df.loc[0, "SA75"])
                        result_dict["msa_8th"].append(df.loc[7, "mSA"])
                        result_dict["sa50_8th"].append(df.loc[7, "SA50"])
                        result_dict["sa75_8th"].append(df.loc[7, "SA75"])
                        result_dict["dataset"].append(dataset)

                df = pd.DataFrame(result_dict)
                if df.empty:
                    continue

                df.to_csv(csv_out, index=False)
                if mode in ["amg", "ais"]:
                    df = df[["dataset", "msa"]]
                    df.rename(columns={"msa": f"{model}_{checkpoint}_{mode}"}, inplace=True)
                else:
                    df = df[["dataset", "msa_1st", "msa_8th"]]
                    df.rename(
                        columns={
                            "msa_1st": f"{model}_{checkpoint}_{mode}",
                            "msa_8th": f"{model}_{checkpoint}_I{mode[0]}"
                        },
                        inplace=True,
                    )

                if concat_df.empty:
                    concat_df = df
                else:
                    concat_df = pd.merge(concat_df, df, on="dataset", how="outer")
                    print(f"{model} (checkpoint: {checkpoint}) added to concat results")

        if not concat_df.empty:
            concat_df.to_csv(csv_concat, index=False)


def main():
    get_results("/mnt/lustre-grete/usr/u12649/models/", overwrite=True)


if __name__ == "__main__":
    main()
