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
        "--reshape_embed_size", type=int, nargs="+", default=(8, 8, 8), 
        help="Reshape size for the embeddings (default: 8 8 8)",
    )
    parser.add_argument(
        "--reshape_crop_size", type=int, nargs="+", default=(128, 128, 64),
        help="Original image crop size (default: 128, 128, 64)",
    )
    parser.add_argument(
        "--image_size", type=int, nargs="+", default=(384, 384, 256), 
        help="Original image size (default: 384 384 256)",
    )
    return parser


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(args):

    reconstructions = Path(args.embedding_dir).glob("valid_*")

    for idx_recon, reconstruction in enumerate(reconstructions):
        embed_path = reconstruction / f"{args.embedding_type}.npy"
        if not embed_path.exists():
            print(f"Embedding file {embed_path} does not exist. Skipping.")
            continue

        embeds = np.load(embed_path)
        assert embeds.ndim == 3, f"Expected 3D embedding, got {embeds.ndim}D for {embed_path}"
        num_crops, num_tokens, embedding_dim = embeds.shape

        expected_tokens = math.prod(args.reshape_embed_size)
        assert num_tokens == expected_tokens, \
            f"Expected {expected_tokens} tokens but got {num_tokens} for reshape size {args.reshape_crop_size}"

        # Flatten all embeddings to fit PCA
        flattened = embeds.reshape(-1, embedding_dim)  # Shape: (num_crops * num_tokens, embedding_dim)

        if idx_recon == 0:
            pca = PCA(n_components=3)
            pca = pca.fit(flattened)  # Fit PCA on the first reconstruction
        
        flattened_pca = pca.transform(flattened)  # Shape: (num_crops * num_tokens, 3)

        means = flattened_pca.mean(axis=0)
        stds = flattened_pca.std(axis=0)

        # Reshape back to (num_crops, reshape_size..., 3)
        flattened_pca = flattened_pca.reshape(num_crops, *args.reshape_embed_size, 3)

        #pca_embeds = []
        #for pca_embedding in flattened_pca:
        normed = (flattened_pca - means) / (stds + 1e-8)
        normed = sigmoid(normed)
        normed = (normed * 255).astype(np.uint8)
        #pca_embeds.append(normed)

        # Combine patches into a full volume
        grid_size = tuple(img_sz // crop_sz for img_sz, crop_sz in zip(
            args.image_size, args.reshape_crop_size))
        
        normed = normed.reshape(*grid_size, *args.reshape_embed_size, 3)
        normed = normed.transpose(0, 3, 1, 4, 2, 5, 6)
        normed = normed.reshape(
            grid_size[0] * args.reshape_embed_size[0],
            grid_size[1] * args.reshape_embed_size[1],
            grid_size[2] * args.reshape_embed_size[2],
            3
        )

        #combined_embeds = combine_patches_numpy(pca_embeds, grid_size)

        # Resize to image size (D, H, W, C)
        zoom_factors = (
            args.image_size[0] / normed.shape[0],
            args.image_size[1] / normed.shape[1],
            args.image_size[2] / normed.shape[2],
            1,
        )
        combined_embeds = zoom(normed, zoom_factors, order=1)

        # Reorder from RAS for visualization
        combined_embeds = np.transpose(combined_embeds, (1, 0, 2, 3))
        combined_embeds = np.flip(combined_embeds, axis=(0, 1))

        image_path = reconstruction / "image.npy"
        if not image_path.exists():
            print(f"Image file {image_path} does not exist. Skipping.")
            continue

        img = np.load(image_path)
        img = img.reshape(*grid_size, *img.shape[1:])
        img = img.transpose(3, 0, 4, 1, 5, 2, 6)
        img = img.reshape(1, *args.image_size)

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
        print(f"Saved PCA gif to {gif_path}")


if __name__ == "__main__":
    
    parser = get_args_parser()
    args = parser.parse_args([
        "--embedding_dir", r"E:\spectre\results\eval\embeddings_ct_rate\dinov2\vit-l (valiant-salad-785)",
        "--embedding_type", "image_backbone_patch",
        "--reshape_embed_size", "8", "8", "8",
        "--reshape_crop_size", "128", "128", "64",
        "--image_size", "512", "512", "384",
    ])
    main(args)
