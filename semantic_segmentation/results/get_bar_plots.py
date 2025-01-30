import os
from glob import glob
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt


COLOR_MAPS = {
    "cellvit": "#12711c",
    "hovernet": "#001c7f",
    "hovernext": "#8c0800",
    "pathosam": "#ed409a",
    "biomedparse": "#006374",
}

MP_MAPS = {
    "hovernet_semantic": "HoVerNet",
    "hovernext_1_semantic": "HoVerNeXt",
    "cellvit_sam_40_semantic": "CellViT",
    "biomedparse_semantic": "BioMedParse",
    "pathosam_finetune_all-from_pretrained": "PathoSAM (Generalist)",
}


def get_main_paper_bars():
    # Below gets results for all methods.
    res_paths = glob(os.path.join("*.csv"))
    all_results = {}
    for res_path in res_paths:
        res = pd.read_csv(res_path)
        weighted_mean = res.iloc[0]["weighted_mean"]
        method_name = Path(res_path).stem
        if method_name not in MP_MAPS:
            continue

        all_results[method_name] = weighted_mean

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar([MP_MAPS[k] for k in all_results.keys()], all_results.values())
    plt.xticks(fontsize=16)
    plt.ylabel('Weighted Dice Similarity Coefficient', fontsize=16)
    plt.title('Semantic Segmentation', fontsize=16)

    plt.savefig("./semantic.svg")
    plt.savefig("./semantic.png")
    plt.close()


def main():
    get_main_paper_bars()


if __name__ == "__main__":
    main()
