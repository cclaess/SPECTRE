import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser(description="Visualize projected image embeddings with t-SNE")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Root directory where embeddings are stored")
    parser.add_argument("--embedding_type", type=str, default="image_projection", help="Which embedding to load (e.g. image_projection)")
    parser.add_argument("--output_plot", type=str, default=None, help="Path to save the plot (optional)")
    return parser


def assign_group(n):
    if n == 0:
        return '0'
    elif 1 <= n <= 4:
        return '1-4'
    elif 5 <= n <= 8:
        return '5-8'
    else:
        return '9+'


def main(args):
    df = pd.read_csv(args.csv_path)
    df_filtered = df[df["VolumeName"].str.endswith("_1.nii.gz")].copy()

    cols_to_check = df_filtered.columns.difference(["VolumeName"])
    df_filtered["num_abnormalities"] = (df_filtered[cols_to_check] == 1).sum(axis=1)
    df_filtered["group"] = df_filtered["num_abnormalities"].apply(assign_group)

    embeddings = []
    labels = []
    missing = []

    for _, row in df_filtered.iterrows():
        filename = row["VolumeName"].split(".")[0]
        emb_path = Path(args.embedding_dir) / filename / f"{args.embedding_type}.npy"

        if emb_path.exists():
            emb = np.load(emb_path)
            if emb.ndim > 1:
                emb = emb.flatten()
            embeddings.append(emb)
            labels.append(row["group"])
        else:
            missing.append(str(emb_path))

    if missing:
        print(f"Warning: {len(missing)} missing embeddings.")
        print("Examples:", missing[:5])

    embeddings = np.array(embeddings)
    print(f"Loaded {len(embeddings)} embeddings.")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(embeddings)

    group_colors = {
        "0": "red",
        "1-4": "orange",
        "5-8": "blue",
        "9+": "green"
    }

    plt.figure(figsize=(10, 8))
    for group in sorted(set(labels)):
        idx = [i for i, g in enumerate(labels) if g == group]
        plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1],
                    label=group, color=group_colors.get(group, "gray"), alpha=0.7)

    plt.title("t-SNE Visualization of Projected Image Embeddings")
    plt.legend(title="Number of abnormalities")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()

    if args.output_plot:
        plt.savefig(args.output_plot)
        print(f"Plot saved to {args.output_plot}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
