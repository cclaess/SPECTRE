import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP


EMBEDDING_NAME_MAP = {
    "image_projection": "image",
    "text_projection": "text",
}


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Visualize projected embeddings with t-SNE"
    )
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to CSV file with VolumeName and abnormality flags"
    )
    parser.add_argument(
        "--embedding_dir", type=str, required=True,
        help="Directory where each sample subfolder holds embedding .npy files"
    )
    parser.add_argument(
        "--embedding_types", type=str, nargs="+", 
        default=["image_projection", "text_projection"],
        help="Which .npy embeddings to load (without extension). " \
        "The first type will be used for fitting the UMAP model."
    )
    parser.add_argument(
        "--n_neighbors", type=int, default=50,
        help="Number of neighbors for UMAP"
    )
    parser.add_argument(
        "--output_plot", type=str, default=None,
        help="Path to save the UMAP plot (optional)"
    )
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

    # Load labels
    df = pd.read_csv(args.csv_path)
    cols = df.columns.difference(["VolumeName"])
    df["num_abnormalities"] = (df[cols] == 1).sum(axis=1)
    df["group"] = df["num_abnormalities"].apply(assign_group)

     # Load embeddings
    embeddings = {etype: [] for etype in args.embedding_types}
    missing = {etype: [] for etype in args.embedding_types}
    valid_rows = {etype: [] for etype in args.embedding_types}

    for etype in args.embedding_types:
        for _, row in df.iterrows():
            fname = str(row["VolumeName"]).replace(".nii.gz", "")
            emb_path = Path(args.embedding_dir) / fname / f"{etype}.npy"
            if not emb_path.exists():
                missing[etype].append(str(emb_path))
                continue
            emb = np.load(emb_path)
            embeddings[etype].append(emb.flatten())
            valid_rows[etype].append(row)

        if missing[etype]:
            print(f"Warning: {len(missing[etype])} embeddings missing for {etype}.")
        
        embeddings[etype] = np.array(embeddings[etype])
        valid_rows[etype] = pd.DataFrame(valid_rows[etype]).reset_index(drop=True)

    # Fit UMAP to first embeddings
    umap = UMAP(n_neighbors=args.n_neighbors, n_components=2, random_state=42)
    umap = umap.fit(embeddings[args.embedding_types[0]])

    # Transform all embeddings with UMAP
    umap_result = {etype: umap.transform(embeddings[etype]) for etype in args.embedding_types}

    # Plot
    group_order = ["0", "1-4", "5-8", "9+"]
    palette = sns.color_palette("coolwarm", len(group_order))
    group_to_color = {group: palette[i] for i, group in enumerate(group_order)}

    num_plots = len(args.embedding_types)
    _, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8), squeeze=False)

    # Get limits for consistent axis scalings
    all_x = np.concatenate([umap_result[etype][:, 0] for etype in args.embedding_types])
    all_y = np.concatenate([umap_result[etype][:, 1] for etype in args.embedding_types])

    x_center = (all_x.min() + all_x.max()) / 2
    x_range = (all_x.max() - all_x.min()) * 0.6
    x_min, x_max = x_center - x_range, x_center + x_range

    y_center = (all_y.min() + all_y.max()) / 2
    y_range = (all_y.max() - all_y.min()) * 0.6
    y_min, y_max = y_center - y_range, y_center + y_range

    for i, etype in enumerate(args.embedding_types):
        ax = axes[0, i]
        df_subset = valid_rows[etype]
        for grp in group_order:
            if grp in df_subset["group"].values:
                idxs = df_subset.index[df_subset["group"] == grp].tolist()
                ax.scatter(
                    umap_result[etype][idxs, 0], umap_result[etype][idxs, 1],
                    label=grp, color=group_to_color.get(grp, "gray"), alpha=1.0, s=7
                )

        display_name = EMBEDDING_NAME_MAP.get(etype, etype)
        ax.set_title(f"UMAP of {display_name} embeddings", fontsize=20)
        ax.set_xlabel("UMAP 1", fontsize=18)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if i == 0:
            ax.set_ylabel("UMAP 2", fontsize=18)
            ax.legend(title="# abnormalities", fontsize=14, title_fontsize=14)
            ax.tick_params(labelsize=14)
        else:
            ax.tick_params(labelleft=False, left=False, labelsize=14)

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
