import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Compute text→image Recall@K within non-overlapping pools (Merlin-style)"
    )
    parser.add_argument(
        "--emb_dir", type=Path, required=True,
        help="Root folder containing one subfolder per sample (basename -> files)"
    )
    parser.add_argument(
        "--txt_emb", type=str, default="text_projection",
        help="Filename (no .npy) of text embeddings"
    )
    parser.add_argument(
        "--img_emb", type=str, default="image_projection",
        help="Filename (no .npy) of image embeddings (candidates)"
    )
    parser.add_argument(
        "--pool_size", type=int, default=128,
        help="Size N of non-overlapping pools to partition the dataset into"
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[1, 8],
        help="Recall@K cutoffs (space separated), e.g. --ks 5 10 50"
    )
    return parser


def load_embeddings(emb_dir: Path, key: str) -> Tuple[np.ndarray, List[str]]:
    """
    For each row in df, load emb_dir / <basename> / (key + '.npy'),
    flatten to 1D, return array of shape (N, D), and list of basenames.
    """
    embs = []
    ids = []
    base_paths = sorted(p for p in emb_dir.glob("*") if p.is_dir())
    for base in base_paths:
        path = base / f"{key}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        e = np.load(path)
        embs.append(e.flatten())
        ids.append(base)
    return np.vstack(embs), ids


def partition_indices(n_samples: int, pool_size: int):
    """
    Generate list of (start, end) index pairs (end exclusive) for non-overlapping pools.
    """
    if pool_size <= 0:
        raise ValueError("pool_size must be > 0")
    idxs = []
    for start in range(0, n_samples, pool_size):
        end = min(start + pool_size, n_samples)
        idxs.append((start, end))
    return idxs


def recall_counts_per_pool(sim_mat: np.ndarray, ks: List[int]) -> dict:
    """
    sim_mat: (M, N) similarity between M text queries and N images in same pool.
             For our use case, M == N (one text per image in pool) but function keeps it general.
    ks: list of int cutoffs
    Returns dict {k: correct_count} where correct_count is the NUMBER of queries
    for which the ground-truth image (index i) appears in top-k for query i.
    Assumes ground-truth for text i is image i (i.e., aligned ordering).
    """
    M, N = sim_mat.shape
    # sort indices descending sim
    ranks = np.argsort(-sim_mat, axis=1)
    counts = {k: 0 for k in ks}
    # ground-truth is same-index (i -> i) provided M == N; if M != N, we only count where i < N
    for i in range(M):
        # only valid if ground truth exists in candidate set
        if i >= N:
            # ground-truth image not present among candidates for this query (shouldn't happen in our pooling)
            continue
        ranklist = ranks[i]
        for k in ks:
            if i in ranklist[:k]:
                counts[k] += 1
    return counts


def main(args):

    # 1) load embeddings
    txt_embs, txt_ids = load_embeddings(Path(args.emb_dir), args.txt_emb)
    img_embs, img_ids = load_embeddings(Path(args.emb_dir), args.img_emb)

    # ensure same ordering
    if txt_ids != img_ids:
        raise AssertionError("Image and text sets must be same samples and in same order")

    # 2) partition into pools
    n_samples = txt_embs.shape[0]
    pools = partition_indices(n_samples, args.pool_size)
    print(f"Dataset contains {n_samples} samples -> {len(pools)} pools (pool_size={args.pool_size})")
    ks = sorted(args.ks)

    # aggregate counts
    total_queries = 0
    total_counts = {k: 0 for k in ks}

    for pool_idx, (start, end) in enumerate(pools, start=1):
        pool_size = end - start
        txt_pool = txt_embs[start:end]
        img_pool = img_embs[start:end]

        # similarity: (M, N) where M == N == pool_size
        sim = cosine_similarity(txt_pool, img_pool)
        counts = recall_counts_per_pool(sim, ks)

        # accumulate
        for k in ks:
            total_counts[k] += counts[k]
        total_queries += pool_size

        # per-pool recalls (fraction)
        pool_recalls = {k: counts[k] / pool_size for k in ks}
        pool_recalls_str = ", ".join([f"R@{k}={pool_recalls[k]:.4f}" for k in ks])
        print(f"Pool {pool_idx:3d} [{start}:{end}] size={pool_size:3d} -> {pool_recalls_str}")

    # final aggregated recall (weighted by queries)
    final_recalls = {k: (total_counts[k] / total_queries) if total_queries > 0 else 0.0 for k in ks}
    print("\n### Aggregated Text→Image Recall@K (pools as candidate sets) ###")
    for k in ks:
        print(f" Recall@{k:3d}: {final_recalls[k]:.4f}")

    # also return results (useful if this function is called programmatically)
    return final_recalls


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
