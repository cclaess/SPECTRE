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
    parser.add_argument(
        "--repeats", type=int, default=100,
        help="Number of times to repeat random non-overlapping pool partitioning (bootstrap repeats)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility of pool partitioning"
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


def partition_indices_from_permutation(perm: np.ndarray, pool_size: int):
    """
    Given a 1D array `perm` of shuffled indices, generate list of (start, end, pool_indices)
    for non-overlapping pools built from perm. end is exclusive, pool_indices = perm[start:end]
    """
    if pool_size <= 0:
        raise ValueError("pool_size must be > 0")
    n = perm.shape[0]
    pools = []
    for start in range(0, n, pool_size):
        end = min(start + pool_size, n)
        pool_indices = perm[start:end]
        pools.append((start, end, pool_indices))
    return pools


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


def compute_recall_for_partition(txt_embs: np.ndarray, img_embs: np.ndarray, pools, ks: List[int],
                                 print_pools: bool = False):
    """
    Given embeddings and a list of pools (each pool is (start, end, pool_indices)),
    compute aggregated recall counts and total_queries, and optionally print per-pool recalls.
    Returns total_counts dict and total_queries.
    """
    total_counts = {k: 0 for k in ks}
    total_queries = 0
    for pool_idx, (start, end, pool_indices) in enumerate(pools, start=1):
        pool_size = end - start
        # aligned selection: both txt and img use the same pool_indices -> ground-truth maps i->i within pool
        txt_pool = txt_embs[pool_indices]
        img_pool = img_embs[pool_indices]

        sim = cosine_similarity(txt_pool, img_pool)
        counts = recall_counts_per_pool(sim, ks)

        for k in ks:
            total_counts[k] += counts[k]
        total_queries += pool_size

        if print_pools:
            pool_recalls = {k: counts[k] / pool_size for k in ks}
            pool_recalls_str = ", ".join([f"R@{k}={pool_recalls[k]:.4f}" for k in ks])
            print(f"Pool {pool_idx:3d} indices[{start}:{end}] size={pool_size:3d} -> {pool_recalls_str}")

    return total_counts, total_queries


def main(args):

    # 1) load embeddings
    txt_embs, txt_ids = load_embeddings(Path(args.emb_dir), args.txt_emb)
    img_embs, img_ids = load_embeddings(Path(args.emb_dir), args.img_emb)

    # ensure same ordering
    if txt_ids != img_ids:
        raise AssertionError("Image and text sets must be same samples and in same order")

    # 2) partition into pools
    n_samples = txt_embs.shape[0]
    ks = sorted(args.ks)
    repeats = max(1, int(args.repeats))
    seed = int(args.seed)

    print(f"Dataset contains {n_samples} samples")
    print(f"Pool size: {args.pool_size}, ks: {ks}, repeats: {repeats}, seed: {seed}")

    # Prepare RNG
    rng = np.random.default_rng(seed)

    # store per-repeat recalls (fraction)
    recalls_per_repeat = {k: [] for k in ks}

    for r in range(repeats):
        # shuffle indices and partition into non-overlapping pools
        perm = np.arange(n_samples)
        rng.shuffle(perm)  # in-place
        pools = partition_indices_from_permutation(perm, args.pool_size)

        # For readability, only print per-pool breakdown for the first repeat
        print_pools = (repeats == 1) or (r == 0)

        total_counts, total_queries = compute_recall_for_partition(
            txt_embs, img_embs, pools, ks, print_pools=print_pools
        )

        # compute recall fractions for this repeat
        if total_queries == 0:
            raise ValueError("No queries found (total_queries == 0)")
        recalls_this = {k: (total_counts[k] / total_queries) for k in ks}

        print(f"Repeat {r+1:3d}/{repeats:3d} -> " +
              ", ".join([f"R@{k}={recalls_this[k]:.4f}" for k in ks]))
        for k in ks:
            recalls_per_repeat[k].append(recalls_this[k])

    # After repeats -> compute mean and 95% percentile CI across repeats
    print("\n### Aggregated Text→Image Recall@K across repeats ###")
    for k in ks:
        arr = np.array(recalls_per_repeat[k])
        mean = arr.mean()
        if repeats > 1:
            lower, upper = np.percentile(arr, [2.5, 97.5])
        else:
            lower, upper = mean, mean
        print(f" Recall@{k:3d}: mean={mean:.4f}  95% CI=[{lower:.4f}, {upper:.4f}]  (n_repeats={repeats})")

    # also return results programmatically
    final_stats = {
        k: {
            "mean": float(np.mean(recalls_per_repeat[k])),
            "ci_lower": float(np.percentile(recalls_per_repeat[k], 2.5)) if repeats > 1 else float(np.mean(recalls_per_repeat[k])),
            "ci_upper": float(np.percentile(recalls_per_repeat[k], 97.5)) if repeats > 1 else float(np.mean(recalls_per_repeat[k])),
            "values": recalls_per_repeat[k],
        } for k in ks
    }
    return final_stats


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
