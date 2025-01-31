import os
from glob import glob
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


MP_MAPS = {
    "hovernet_semantic": "HoVerNet",
    "hovernext_1_semantic": "HoVerNeXt",
    "cellvit_sam_40_semantic": "CellViT",
    "biomedparse_semantic": "BioMedParse",
    "pathosam_finetune_all-from_scratch": r"$\bf{PathoSAM}$" + "\n" + r"$\bf{(Generalist)}$",
}


COLOR_MAPS = {
    "cellvit": "#12711c",
    "hovernet": "#001c7f",
    "hovernext": "#8c0800",
    "pathosam": "#b8850a",
    "biomedparse": "#006374",
}

YTICKS = {
    "hovernet": "HoVerNet",
    "hovernext": "HoverNeXt",
    "cellvit": "CellViT",
    "biomedparse": "BioMedParse",
    "pathosam": r"$\bf{PathoSAM}$" + "\n" + r"$\bf{(Generalist)}$",
}

SUPP_MAPS_PER_CLASS = {
    "hovernet_semantic": "HoVerNet",
    "hovernext_1_semantic": "HoVerNeXt\n(PanNuke-ConvNeXt-1)",
    "hovernext_2_semantic": "HoVerNeXt\n(PanNuke-ConvNeXt-2)",
    "cellvit_256_20_semantic": "CellViT\n(256-x20)",
    "cellvit_256_40_semantic": "CellViT\n(256-x40)",
    "cellvit_sam_20_semantic": "CellViT\n(SAM-H-x20)",
    "cellvit_sam_40_semantic": "CellViT\n(SAM-H-x40)",
    "biomedparse_semantic": "BioMedParse",
    "pathosam_finetune_decoder_only-from_pretrained": r"PathoSAM" + "\n" + r"(IE$_{\text{Freeze}}$" + " + " + r"SD$_{\text{FT}}$)",  # noqa
    "pathosam_finetune_decoder_only-from_scratch": r"PathoSAM" + "\n" + r"(IE$_{\text{Freeze}}$" + " + " + r"SD$_{\text{Scratch}}$)",  # noqa
    "pathosam_finetune_all-from_pretrained": r"PathoSAM" + "\n" + r"(IE$_{\text{FT}}$" + " + " + r"SD$_{\text{FT}}$)",
    "pathosam_finetune_all-from_scratch": r"PathoSAM" + "\n" + r"(IE$_{\text{FT}}$" + " + " + r"SD$_{\text{Scratch}}$)",
}

SUPP_MAPS = {
    "hovernet_semantic": "HoVerNet",
    "hovernext_1_semantic": "PanNuke-ConvNeXt-1",
    "hovernext_2_semantic": "PanNuke-ConvNeXt-2",
    "cellvit_256_20_semantic": "256-x20",
    "cellvit_256_40_semantic": "256-x40",
    "cellvit_sam_20_semantic": "SAM-H-x20",
    "cellvit_sam_40_semantic": "SAM-H-x40",
    "biomedparse_semantic": "BioMedParse",
    "pathosam_finetune_decoder_only-from_pretrained": r"IE$_{\text{Freeze}}$" + " + " + r"SD$_{\text{FT}}$",
    "pathosam_finetune_decoder_only-from_scratch": r"IE$_{\text{Freeze}}$" + " + " + r"SD$_{\text{Scratch}}$",
    "pathosam_finetune_all-from_pretrained": r"IE$_{\text{FT}}$" + " + " + r"SD$_{\text{FT}}$",
    "pathosam_finetune_all-from_scratch": r"IE$_{\text{FT}}$" + " + " + r"SD$_{\text{Scratch}}$",
}

CLASS_COLORS = {
    "neoplastic_cells": "#E69F00",
    "inflammatory_cells": "#56B4E9",
    "connective_cells": "#009E73",
    "dead_cells": "#F0E442",
    "epithelial_cells": "#CC79A7",
}


CLASS_NAME_MAP = {
    "neoplastic_cells": "Neoplastic Cells",
    "inflammatory_cells": "Inflammatory Cells",
    "connective_cells": "Connective Tissue",
    "dead_cells": "Dead Cells",
    "epithelial_cells": "Epithelial Cells",
}


def get_appendix_per_class_semantic_plot():
    res_paths = glob(os.path.join("*.csv"))
    all_results = OrderedDict()

    for res_path in res_paths:
        res = pd.read_csv(res_path)
        unnamed_cols = [col for col in res.columns if "Unnamed" in col]
        if unnamed_cols:
            res = res.drop(columns=unnamed_cols)

        method_name = Path(res_path).stem
        if method_name in SUPP_MAPS_PER_CLASS:
            all_results[method_name] = res.iloc[0]

    ordered_results = OrderedDict((key, all_results[key]) for key in SUPP_MAPS.keys() if key in all_results)

    all_columns = list(next(iter(ordered_results.values())).index)
    columns_to_plot = [col for col in all_columns if col not in ["weighted_mean", "absolute_mean", "0"]]

    num_groups = len(ordered_results)
    num_bars_per_group = len(columns_to_plot)
    group_width = 0.9
    bar_width = group_width / num_bars_per_group

    group_labels = [SUPP_MAPS_PER_CLASS[k] for k in ordered_results.keys()]
    x = np.linspace(0, num_groups * 1.5, num_groups)

    plt.figure(figsize=(30, 15))

    for j, col in enumerate(columns_to_plot):
        offsets = np.linspace(-group_width / 2, group_width / 2, num_bars_per_group)
        for i, (method, row) in enumerate(ordered_results.items()):
            color = CLASS_COLORS.get(col, "#333333")

            plt.bar(
                x[i] + offsets[j], row[col], color=color, width=bar_width,
                label=CLASS_NAME_MAP.get(col, col) if i == 0 else None, edgecolor="white"
            )

    plt.xticks(x, group_labels, fontsize=14, rotation=15, ha="center")
    plt.yticks(fontsize=14)
    plt.ylabel("Dice Score Coefficient", fontsize=18, fontweight="bold")
    plt.title("Semantic Segmentation (PanNuke: Per-Class Absolute Score)", fontsize=18, fontweight="bold")

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = {CLASS_NAME_MAP.get(label, label): handle for label, handle in zip(labels, handles)}

    plt.legend(
        unique_labels.values(), unique_labels.keys(), fontsize=14,
        loc="upper left", bbox_to_anchor=(0.25, 1), ncol=len(columns_to_plot),
    )

    plt.tight_layout()
    plt.savefig("./semantic_quanti_all_per_class.svg")
    plt.savefig("./semantic_quanti_all_per_class.png")
    plt.close()


def get_appendix_semantic_plots(mean_type="weighted_mean"):
    res_paths = glob(os.path.join("*.csv"))
    all_results = {}

    for res_path in res_paths:
        res = pd.read_csv(res_path)
        method_name = Path(res_path).stem
        all_results[method_name] = res.iloc[0][mean_type]

    grouped_results = {}
    for key in SUPP_MAPS.keys():
        if key in all_results:
            prefix = key.split("_")[0]
            if prefix not in grouped_results:
                grouped_results[prefix] = []
            grouped_results[prefix].append((SUPP_MAPS[key], all_results[key]))

    group_width = 0.8
    bar_width = group_width / max(len(v) for v in grouped_results.values())

    group_labels = list(grouped_results.keys())
    yticks_labels = [YTICKS.get(label, label) for label in group_labels]
    x = np.arange(len(group_labels))

    plt.figure(figsize=(15, 10))

    for i, (prefix, methods) in enumerate(grouped_results.items()):
        offsets = np.linspace(-bar_width * (len(methods) - 1) / 2, bar_width * (len(methods) - 1) / 2, len(methods))

        for (method_label, value), offset in zip(methods, offsets):
            color = COLOR_MAPS.get(prefix, "#333333")
            plt.bar(
                x[i] + offset, value, color=color, width=bar_width,
                label=method_label if i == 0 else None, edgecolor="white", linewidth=2,
            )

            if len(methods) > 1:
                plt.text(
                    x[i] + offset + 0.05, 0.01, method_label, fontsize=16, fontweight="bold",
                    color="white", va="bottom", rotation=90, wrap=True
                )

    plt.xticks(x, yticks_labels, fontsize=16, ha="center")
    plt.ylabel("Mean Dice Similarity Coefficient", fontsize=16, fontweight="bold")
    mname = "Weighted Mean Scores" if mean_type == "weighted_mean" else "Absolute Mean Scores"
    plt.title(f"Semantic Segmentation (PanNuke: {mname})", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"./semantic_quanti_all_appedix-{mean_type}.svg")
    plt.savefig(f"./semantic_quanti_all_appendix-{mean_type}.png")
    plt.close()


def get_main_paper_plots():
    res_paths = glob(os.path.join("*.csv"))
    all_results = {}

    for res_path in res_paths:
        res = pd.read_csv(res_path)
        weighted_mean = res.iloc[0]["weighted_mean"]
        method_name = Path(res_path).stem
        if method_name in MP_MAPS:
            all_results[method_name] = weighted_mean

    method_labels = []
    method_values = []
    colors = []

    for key in MP_MAPS.keys():
        if key in all_results:
            method_labels.append(MP_MAPS[key])
            method_values.append(all_results[key])
            color = next((c for p, c in COLOR_MAPS.items() if key.startswith(p)), "#333333")
            colors.append(color)

    plt.figure(figsize=(10, 6))
    plt.bar(method_labels, method_values, color=colors, width=0.4)

    plt.xticks(fontsize=10, color="#262626")
    plt.yticks(fontsize=10, color="#262626")
    plt.ylabel("Weighted-Mean Dice Similarity Coefficient", fontsize=16, fontweight="bold", color="#262626")
    plt.title("Semantic Segmentation (PanNuke)", fontsize=16, fontweight="bold", color="#262626")

    ax = plt.gca()
    ax.spines["top"].set_color("#262626")
    ax.spines["right"].set_color("#262626")
    ax.spines["bottom"].set_color("#262626")
    ax.spines["left"].set_color("#262626")

    plt.grid(False)

    plt.tick_params(axis="x", length=0)
    plt.tick_params(axis="y", length=0)

    def custom_format(x, _):
        return "0" if x == 0 else f"{x:.1f}"

    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_format))

    pathosam_label = MP_MAPS.get("pathosam_finetune_all-from_scratch", None)
    if pathosam_label and pathosam_label in method_labels:
        pathosam_index = method_labels.index(pathosam_label)
        pathosam_value = method_values[pathosam_index]
        pathosam_color = colors[pathosam_index]

        plt.axhline(y=pathosam_value - 0.0025, color=pathosam_color, linestyle="dashed", linewidth=3, zorder=0)

    plt.tight_layout()
    plt.savefig("./semantic_quanti_fig_3.svg")
    plt.savefig("./semantic_quanti_fig_3.png")
    plt.close()


def main():
    get_main_paper_plots()
    get_appendix_semantic_plots("weighted_mean")
    get_appendix_semantic_plots("absolute_mean")
    get_appendix_per_class_semantic_plot()


if __name__ == "__main__":
    main()
