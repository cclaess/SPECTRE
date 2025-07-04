import math
import argparse
from pathlib import Path

import imageio
import numpy as np
from scipy.ndimage import zoom
from sklearn.decomposition import PCA


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Visualize projected image embeddings with t-SNE"
    )
    parser.add_argument(
        "--embedding_dir", type=str, required=True, 
        help="Root directory where embeddings are stored",
    )
    parser.add_argument(
        "--embedding_type", type=str, default="image_backbone_full", 
        help="Which embedding to load (e.g. image_backbone_full)",
    )
    parser.add_argument(
        "--reshape_size", type=int, nargs="+", default=(8, 8, 8), 
        help="Reshape size for the embeddings (default: 8 8 8)",
    )
    parser.add_argument(
        "--image_size", type=int, nargs="+", default=(384, 384, 256), 
        help="Original image size (default: 384 384 256)",
    )
    return parser


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def combine_patches_numpy(patches: list[np.ndarray], grid_size: tuple[int, int, int]) -> np.ndarray:
    """
    Combine 3D patches with embeddings into a full volume using NumPy.

    Args:
        patches: list of np.ndarray, each with shape (8, 8, 8, embed_dim)
        grid_size: tuple (num_x, num_y, num_z)

    Returns:
        np.ndarray: combined array of shape (X*8, Y*8, Z*8, embed_dim)
    """
    num_x, num_y, num_z = grid_size
    d, h, w, c = patches[0].shape

    full_shape = (d * num_x, h * num_y, w * num_z, c)
    combined = np.zeros(full_shape, dtype=patches[0].dtype)

    idx = 0
    for z in range(num_z):
        for y in range(num_y):
            for x in range(num_x):
                combined[
                    x * d: (x + 1) * d,
                    y * h: (y + 1) * h,
                    z * w: (z + 1) * w,
                    :
                ] = patches[idx]
                idx += 1

    return combined


def main(args):

    reconstructions = Path(args.embedding_dir).glob("valid_*")

    for reconstruction in reconstructions:
        embed_path = reconstruction / f"{args.embedding_type}.npy"
        if not embed_path.exists():
            print(f"Embedding file {embed_path} does not exist. Skipping.")
            continue

        embeds = np.load(embed_path)
        assert embeds.ndim == 3, f"Expected 3D embedding, got {embeds.ndim}D for {embed_path}"
        num_crops, num_tokens, embedding_dim = embeds.shape

        expected_tokens = math.prod(args.reshape_size)
        assert num_tokens == expected_tokens, \
            f"Expected {expected_tokens} tokens but got {num_tokens} for reshape size {args.reshape_size}"

        # Flatten all embeddings to fit PCA
        flattened = embeds.reshape(-1, embedding_dim)  # Shape: (num_crops * num_tokens, embedding_dim)

        pca = PCA(n_components=3)
        flattened_pca = pca.fit_transform(flattened)  # Shape: (num_crops * num_tokens, 3)

        means = flattened_pca.mean(axis=0)
        stds = flattened_pca.std(axis=0)

        # Reshape back to (num_crops, reshape_size..., 3)
        flattened_pca = flattened_pca.reshape(num_crops, *args.reshape_size, 3)

        pca_embeds = []
        for pca_embedding in flattened_pca:
            normed = (pca_embedding - means) / (stds + 1e-8)
            normed = sigmoid(normed)
            normed = (normed * 255).astype(np.uint8)
            pca_embeds.append(normed)

        # Estimate grid size from num_crops (assumes cubic-ish layout)
        grid_x = grid_y = grid_z = round(num_crops ** (1/3))
        if grid_x * grid_y * grid_z < num_crops:
            grid_z += 1
        grid_size = (grid_x, grid_y, grid_z)

        combined_embeds = combine_patches_numpy(pca_embeds, grid_size)

        # Resize to image size (D, H, W, C)
        zoom_factors = (
            args.image_size[0] / combined_embeds.shape[0],
            args.image_size[1] / combined_embeds.shape[1],
            args.image_size[2] / combined_embeds.shape[2],
            1,
        )
        combined_embeds = zoom(combined_embeds, zoom_factors, order=1)

        # Reorder from RAS for visualization
        combined_embed = np.transpose(combined_embeds, (1, 0, 2, 3))
        combined_embeds = np.flip(combined_embed, axis=0)
        combined_embeds = np.flip(combined_embeds, axis=1)

        # Create gif frames per-slice
        frames = []
        for i in range(combined_embeds.shape[2]):
            slice_rgb = combined_embeds[:, :, i, :]  # (H, W, 3)
            frames.append(slice_rgb)

        gif_path = reconstruction / "pca_embedding.gif"
        imageio.mimsave(gif_path, frames, duration=0.1)
        print(f"Saved PCA gif to {gif_path}")


if __name__ == "__main__":
    
    parser = get_args_parser()
    args = parser.parse_args([
        "--embedding_dir", r"E:\spectre\results\eval\embeddings_ct_rate\vit-b dino",
    ])
    main(args)
