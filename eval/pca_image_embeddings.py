import math
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
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
        "--embedding_type", type=str, default="image_backbone_patch", 
        help="Which embedding to load (e.g. image_backbone_patch)",
    )
    parser.add_argument(
        "--patch_grid_size", type=int, nargs="+", default=(8, 8, 8), 
        help="Reshape size for the embeddings (default: 8 8 8)",
    )
    return parser


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(args):

    reconstructions = sorted(p for p in Path(args.embedding_dir).glob("*") if p.is_dir())

    for reconstruction in reconstructions:
        embed_path = reconstruction / f"{args.embedding_type}.npy"
        if not embed_path.exists():
            print(f"Embedding file {embed_path} does not exist. Skipping.")
            continue

        embeds = np.load(embed_path)
        assert embeds.ndim == 5, f"Expected 5D embedding, got {embeds.ndim}D for {embed_path}"
        Hp, Wp, Dp, num_tokens, embedding_dim = embeds.shape

        expected_tokens = math.prod(args.patch_grid_size)
        assert num_tokens == expected_tokens, \
            f"Expected {expected_tokens} tokens but got {num_tokens}"
        
        image_path = reconstruction / "image.npy"
        if not image_path.exists():
            print(f"Image file {image_path} does not exist. Skipping.")
            continue
        img = np.load(image_path)

        # Flatten all embeddings to fit PCA
        flattened = embeds.reshape(-1, embedding_dim)  # Shape: (num_crops * num_tokens, embedding_dim)

        pca = PCA(n_components=3)
        pca = pca.fit(flattened)  # Fit PCA on the current reconstruction
        
        flattened_pca = pca.transform(flattened)  # Shape: (num_crops * num_tokens, 3)

        means = flattened_pca.mean(axis=0)
        stds = flattened_pca.std(axis=0)

        normed = (flattened_pca - means) / (stds + 1e-8)
        normed = sigmoid(normed)
        normed = (normed * 255).astype(np.uint8)

        normed = normed.reshape(Hp, Wp, Dp, *args.patch_grid_size, 3)
        normed = normed.transpose(0, 3, 1, 4, 2, 5, 6)
        normed = normed.reshape(
            Hp * args.patch_grid_size[0],
            Wp * args.patch_grid_size[1],
            Dp * args.patch_grid_size[2],
            3
        )

        # Resize to image size (D, H, W, C)
        zoom_factors = (
            img.shape[1] / normed.shape[0],
            img.shape[2] / normed.shape[1],
            img.shape[3] / normed.shape[2],
            1,
        )
        combined_embeds = zoom(normed, zoom_factors, order=1)

        # Reorder from RAS for visualization
        combined_embeds = np.transpose(combined_embeds, (1, 0, 2, 3))
        combined_embeds = np.flip(combined_embeds, axis=(0, 1))

        img = np.transpose(img, (2, 1, 3, 0))
        img = np.flip(img, axis=(0, 1))
        img = (img * 255).astype(np.uint8)  # Convert to uint8 for visualization

        # Create gif frames per-slice
        frames = []
        for i in range(combined_embeds.shape[2]):
            slice_rgb = combined_embeds[:, :, i, :]  # (H, W, 3)
            frames.append(slice_rgb)

        gif_path = reconstruction / "pca_embedding.gif"
        frames = [Image.fromarray(frame) for frame in frames]
        frames = [im.convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.NONE) \
                  for im in frames]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            loop=10,
            duration=100,  # 100 ms per frame
            optimize=False,
            disposal=2,  # Background color is replaced by the next frame
        )
        print(f"Saved PCA gif to {gif_path}")

        frames = []
        for i in range(img.shape[2]):
            slice_l = np.repeat(img[:, :, i, :], repeats=3, axis=2)  # (H, W, 3)
            frames.append(slice_l)

        gif_path = reconstruction / "image.gif"
        frames = [Image.fromarray(frame) for frame in frames]
        frames = [im.convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.NONE) \
                  for im in frames]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            loop=10,
            duration=100,  # 100 ms per frame
            optimize=False,
            disposal=2,  # Background color is replaced by the next frame
        )
        print(f"Saved CT gif to {gif_path}")


if __name__ == "__main__":
    
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
