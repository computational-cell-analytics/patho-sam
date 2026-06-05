import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import imagesize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from thresholding_downsampling import run_pipeline
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

CSV_PATH = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/cancerscout_metadata/cancerscout_organized.csv"
ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data"
DARK_AREA_SAMPLES = ["A2020-001401_1-1-1_HE-2021-10-11T16-57-55"]

PCA_COMPONENTS = 3
TILE_SIZE = 224
STRIDE = 224
BATCH_SIZE = 16


# CONFIGURATION
# ─────────────────────────────────────────
PATCH_SIZE = 224
STRIDE = 224
MIN_TISSUE = 0.5
BATCH_SIZE = 64
AGG_METHOD = "mean"  # "mean" | "median" | "mean+std"
N_CLUSTERS = 6
PCA_DIM = 64
OUT_DIR = Path("output")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────
# 1. PATCH EXTRACTION
# ─────────────────────────────────────────


def extract_patches(
    img_rgb: np.ndarray,  # (H, W, 3)  uint8
    mask: np.ndarray,  # (H, W)     uint8 binary
    transform: transforms.Compose,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    min_tissue: float = MIN_TISSUE,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Tiles the downsampled WSI into patches and filters by tissue coverage.

    Returns
    -------
    patches : (N, 3, patch_size, patch_size)  float tensor
    coords  : list of (row, col) top-left corners
    """
    H, W = img_rgb.shape[:2]
    patches, coords = [], []

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            mask_patch = mask[r : r + patch_size, c : c + patch_size]
            if mask_patch.mean() / 255.0 < min_tissue:
                continue
            patch = img_rgb[r : r + patch_size, c : c + patch_size]
            patches.append(transform(Image.fromarray(patch)))
            coords.append((r, c))

    if len(patches) == 0:
        return torch.empty(0), []

    patches_tensor = torch.stack(patches)  # (N, 3, 224, 224)
    print(f"  [patches] {len(coords)} patches extracted from {H}×{W} image")
    return patches_tensor, coords


# ─────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ─────────────────────────────────────────
def extract_features(
    model: torch.nn.Module,
    patches: torch.Tensor,  # (N, 3, 224, 224)
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
) -> np.ndarray:
    """
    Runs patches through UNI v2 and returns feature matrix.

    Returns
    -------
    features : (N, 1536)  float32 numpy array
    """
    loader = DataLoader(TensorDataset(patches), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_feats = []

    with torch.inference_mode():
        for (batch,) in tqdm(loader, desc="  [features]", leave=False):
            feats = model(batch.to(device))  # (B, 1536)
            all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)  # (N, 1536)


# ─────────────────────────────────────────
# 3. AGGREGATION
# ─────────────────────────────────────────
def aggregate_features(
    features: np.ndarray,  # (N_patches, 1536)
    method: str = AGG_METHOD,
) -> np.ndarray:
    """
    Collapses all patch features of one WSI into a single vector.

    Returns
    -------
    wsi_vec : (1536,) or (3072,) depending on method
    """
    if method == "mean":
        return features.mean(axis=0)

    elif method == "median":
        return np.median(features, axis=0)

    elif method == "mean+std":
        return np.concatenate([features.mean(axis=0), features.std(axis=0)])

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ─────────────────────────────────────────
# 4. CLUSTERING
# ─────────────────────────────────────────
def cluster_wsis(
    wsi_features: np.ndarray,  # (N_wsis, feat_dim)
    n_clusters: int = N_CLUSTERS,
    pca_dim: int = PCA_DIM,
    random_state: int = 42,
) -> Tuple[np.ndarray, MiniBatchKMeans]:
    """
    Optionally reduces dimensionality with PCA, then runs K-Means.

    Returns
    -------
    labels : (N_wsis,)
    kmeans : fitted MiniBatchKMeans object
    """
    X = wsi_features.copy()

    # ── L2 normalize ──
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-8)

    # ── StandardScaler ──
    X = StandardScaler().fit_transform(X)

    # ── PCA ──
    if pca_dim and pca_dim < X.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        X = pca.fit_transform(X)
        print(
            f"  [PCA] {wsi_features.shape[1]} → {pca_dim} dims  "
            f"({pca.explained_variance_ratio_.sum():.1%} variance kept)"
        )

    # ── K-Means ──
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=min(4096, len(X)), n_init=20)
    labels = kmeans.fit_predict(X)

    print(f"  [kmeans] {n_clusters} clusters  inertia={kmeans.inertia_:.3e}")
    for k in range(n_clusters):
        n = (labels == k).sum()
        print(f"    Cluster {k}: {n} WSIs  ({n / len(labels):.1%})")

    return labels, kmeans


# ─────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────
def plot_results(
    wsi_features: np.ndarray,
    labels: np.ndarray,
    wsi_names: List[str],
    metadata: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
    out_dir: Path = OUT_DIR,
):
    perplexity = min(30, len(wsi_features) - 1)
    coords_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000).fit_transform(wsi_features)

    # ── build a results df with coordinates ──
    plot_df = pd.DataFrame(
        {"wsi": wsi_names, "cluster": labels, "tsne_1": coords_2d[:, 0], "tsne_2": coords_2d[:, 1]}
    ).set_index("wsi")

    # ── join metadata ──
    plot_df = plot_df.join(metadata, how="left")
    # now plot_df has: cluster, tsne_1, tsne_2, subtype, split, ...

    # ─────────────────────────────────────────
    # PLOT 1: colored by K-Means cluster
    # ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    cmap_cluster = plt.cm.get_cmap("tab10", n_clusters)
    ax = axes[0]
    for k in range(n_clusters):
        sel = plot_df["cluster"] == k
        ax.scatter(
            plot_df.loc[sel, "tsne_1"],
            plot_df.loc[sel, "tsne_2"],
            c=[cmap_cluster(k)],
            label=f"Cluster {k}  (n={sel.sum()})",
            s=80,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.3,
        )
    for wsi_name, row in plot_df.iterrows():
        ax.annotate(wsi_name, (row["tsne_1"], row["tsne_2"]), fontsize=5, alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Colored by K-Means cluster")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # ─────────────────────────────────────────
    # PLOT 2: colored by subtype
    # ─────────────────────────────────────────
    subtypes = ["lepidic", "acinous", "papillary", "micropapillary", "solid", "mucinous"]
    cmap_subtype = plt.cm.get_cmap("Set2", len(subtypes))
    ax = axes[1]

    for i, subtype in enumerate(subtypes):
        if "subtype" not in plot_df.columns:
            break
        sel = plot_df["subtype"] == subtype
        if sel.sum() == 0:
            continue
        ax.scatter(
            plot_df.loc[sel, "tsne_1"],
            plot_df.loc[sel, "tsne_2"],
            c=[cmap_subtype(i)],
            label=f"{subtype}  (n={sel.sum()})",
            s=80,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.3,
        )
    for wsi_name, row in plot_df.iterrows():
        ax.annotate(wsi_name, (row["tsne_1"], row["tsne_2"]), fontsize=5, alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Colored by subtype")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plt.suptitle("WSI-level clustering (t-SNE projection)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "wsi_cluster_tsne.png", dpi=300)
    plt.close()
    print(f"  [plot] saved → {out_dir / 'wsi_cluster_tsne.png'}")

    # ── also save the full plot_df for further analysis ──
    plot_df.to_csv(out_dir / "wsi_tsne_coords.csv")


# ─────────────────────────────────────────
# 6. FULL PIPELINE
# ─────────────────────────────────────────
def run_clustering_pipeline(
    wsi_inputs: List[Tuple[str, Path, Path]],
    model: torch.nn.Module,
    transform: transforms.Compose,
    metadata: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
    agg_method: str = AGG_METHOD,
    pca_dim: int = PCA_DIM,
    device: str = DEVICE,
    out_dir: Path = OUT_DIR,
    cache_dir: Path = None,  # None → defaults to out_dir / "feature_cache"
) -> pd.DataFrame:
    """
    Parameters
    ----------
    wsi_inputs : list of (wsi_name, img_path, mask_path)
    cache_dir  : directory where per-WSI feature vectors are cached as .npy
                 if a cache file exists for a WSI, feature extraction is skipped
    """
    out_dir.mkdir(exist_ok=True)
    cache_dir = cache_dir or out_dir / "feature_cache"
    cache_dir.mkdir(exist_ok=True)

    model.eval()
    model.to(device)

    wsi_names = []
    wsi_features = []

    # ── per-WSI: load → extract → aggregate → cache ──
    for wsi_name, img_path, mask_path in wsi_inputs:
        print(f"\n{'─' * 50}")
        print(f"WSI: {wsi_name}")

        cache_path = cache_dir / f"{wsi_name}_{agg_method}.npy"

        # ── load from cache if available ──
        if cache_path.exists():
            wsi_vec = np.load(cache_path)
            print(f"  [cache] loaded {wsi_vec.shape} from {cache_path.name}")
            wsi_names.append(wsi_name)
            wsi_features.append(wsi_vec)
            continue

        # ── otherwise: full extraction ──
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        print(f"  [load] img={img_rgb.shape}  mask={mask.shape}  ({img_rgb.nbytes / 1e6:.1f} MB)")

        patches, coords = extract_patches(img_rgb, mask, transform)

        if len(coords) == 0:
            print("  [WARN] no tissue patches found — skipping")
            continue

        features = extract_features(model, patches, device=device)
        # (N_patches, 1536)

        wsi_vec = aggregate_features(features, method=agg_method)
        # (1536,)

        # ── save to cache ──
        np.save(cache_path, wsi_vec)
        print(f"  [cache] saved → {cache_path.name}")

        wsi_names.append(wsi_name)
        wsi_features.append(wsi_vec)

        # ── free memory ──
        del img_rgb, mask, patches, features
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"  [OK] {len(coords)} patches → aggregated to {wsi_vec.shape}")

    if len(wsi_features) == 0:
        raise RuntimeError("No valid WSIs found — check your paths/masks.")

    wsi_feature_matrix = np.stack(wsi_features)
    print(f"\n{'─' * 50}")
    print(f"[INFO] Feature matrix: {wsi_feature_matrix.shape}")

    # ── cluster ──
    labels, _ = cluster_wsis(wsi_feature_matrix, n_clusters=n_clusters, pca_dim=pca_dim)

    # ── save csv ──
    results = pd.DataFrame({"wsi": wsi_names, "cluster": labels})
    results.to_csv(out_dir / "wsi_clusters.csv", index=False)
    print(f"\n[OK] Cluster assignments:\n{results.to_string(index=False)}")

    # ── visualize ──
    plot_results(wsi_feature_matrix, labels, wsi_names, n_clusters=n_clusters, out_dir=out_dir, metadata=metadata)

    return results


def get_model_path(model_folder="/mnt/vast-nhr/projects/cidas/cca/models/univ2"):
    filename = "pytorch_model.bin"
    model_path = os.path.join(model_folder, filename)
    if os.path.exists(model_path):
        return model_path


def get_uni_model_and_transform(device):
    model_path = get_model_path()
    model = timm.create_model(
        pretrained=False,
        model_name="vit_giant_patch14_224",
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    model.to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()
    return model, transform


def get_wsi_paths(args, model, transform, target_mpp: float = 2.0):
    df = pd.read_csv(args.csv_path, index_col="filename")
    input_path = Path(args.input_root)
    output_dir = input_path / "univ2" / f"output_mpp_{int(target_mpp)}"
    output_dir.mkdir(exist_ok=True, parents=True)
    sample_dict = {}
    for subtype in ["lepidic", "acinous", "papillary", "micropapillary", "solid", "mucinous"]:
        subtype_df = df[(df[subtype] >= 90) & (df["train_eval_split"].isin(["eval", "train"]))]
        subtype_df_sampled = subtype_df.sample(n=5, random_state=42)
        filename_split_dict = dict(
            zip(subtype_df_sampled.index.tolist(), subtype_df_sampled["train_eval_split"].tolist())
        )
        sample_dict[subtype] = filename_split_dict

    all_wsi_paths = []

    all_patient_ids = [sample[0].split("-")[1].split("_")[0][2:] for sample in all_wsi_paths]
    all_non_tumor_samples = [
        p.stem
        for p in list(input_path.glob("*_models/CancerScout_Lung/new_non_tumor/*.tiff"))
        if p.stem.split("-")[1].split("_")[0][2:] not in all_patient_ids
    ]
    df = df.set_index("patient ID")
    random.seed(42)
    non_tumor_samples = random.sample(all_non_tumor_samples, k=5)
    sample_dict["non_tumor"] = dict(
        zip(
            non_tumor_samples,
            df.loc[[int(p.split("-")[1].split("_")[0][2:]) for p in non_tumor_samples], "train_eval_split"].tolist(),
        )
    )
    metadata_dict = {}
    for subtype, subtype_list in sample_dict.items():
        for filename, split in tqdm(subtype_list.items()):
            entity = "new_non_tumor" if subtype == "non_tumor" else "new_tumor"
            wsi_path = input_path / f"{split}_models" / "CancerScout_Lung" / entity / (filename + ".tiff")
            assert wsi_path.exists(), wsi_path
            file_output_dir: Path = output_dir / filename
            file_output_dir.mkdir(exist_ok=True, parents=True)
            thumbnail_path = file_output_dir / "thumbnail.png"
            mask_path = file_output_dir / "tissue_mask.png"
            if not thumbnail_path.exists():
                img, _ = run_pipeline(wsi_path, output_path=file_output_dir, target_mpp=target_mpp)
                img_shape = img.shape[:2]
            else:
                img_shape = imagesize.get(thumbnail_path)
            sample_dict[subtype][filename] = [split, img_shape, str(file_output_dir)]
            assert thumbnail_path.exists() and mask_path.exists()
            all_wsi_paths.append((filename, thumbnail_path, mask_path))
            metadata_dict[filename] = subtype

    with open(output_dir / "sample_overview.json", "w") as f:
        json.dump(sample_dict, f, indent=4)

    metadata_df = pd.DataFrame(metadata_dict.items(), columns=["filename", "subtype"]).set_index("filename")

    run_clustering_pipeline(
        wsi_inputs=all_wsi_paths,
        model=model,
        transform=transform,
        n_clusters=len(metadata_df["subtype"].unique()),
        device="cuda",
        out_dir=input_path / "univ2" / f"mpp_{int(target_mpp)}" / "clustering",
        cache_dir=input_path / "univ2" / f"mpp_{int(target_mpp)}" / "cached_features",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--csv_path", default=CSV_PATH)
    parser.add_argument("--input_root", "-i", default=ROOT)
    parser.add_argument("--mpp", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = get_uni_model_and_transform(device=device)
    get_wsi_paths(args, model, transform, target_mpp=float(args.mpp))


if __name__ == "__main__":
    main()
