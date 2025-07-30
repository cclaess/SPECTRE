import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Compute image→image MAP@K and text→image Recall@K"
    )
    parser.add_argument(
        "--csv_path", type=Path, required=True,
        help="CSV with VolumeName and one column per abnormality (0/1)"
    )
    parser.add_argument(
        "--emb_dir", type=Path, required=True,
        help="Root folder containing one subfolder per sample"
    )
    parser.add_argument(
        "--img_emb", type=str, default="image_projection",
        help="Filename (no .npy) of image embeddings"
    )
    parser.add_argument(
        "--txt_emb", type=str, default="text_projection",
        help="Filename (no .npy) of text embeddings"
    )
    return parser


def load_embeddings(df, emb_dir: Path, key: str):
    """
    For each row in df, load emb_dir / <basename> / (key + '.npy'),
    flatten to 1D, return array of shape (N, D), and list of basenames.
    """
    embs = []
    ids = []
    for fn in df["VolumeName"]:
        base = str(fn).replace(".nii.gz", "")
        path = emb_dir / base / f"{key}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        e = np.load(path)
        embs.append(e.flatten())
        ids.append(base)
    return np.vstack(embs), ids


def build_label_tuples(df):
    """
    Returns a list of tuples, each tuple is the binary vector of all abnormality columns.
    """
    cols = [c for c in df.columns if c != "VolumeName"]
    return [tuple(row) for row in df[cols].values]


def map_at_k(sim_mat, positives, ks):
    """
    sim_mat: (N, N) similarity matrix (cosine), row i = query i vs all items
    positives: list of sets, positives[i] = set of indices j that are correct matches for i
    ks: list of cutoff ranks
    Returns dict {k: MAP@k}
    """
    N = sim_mat.shape[0]
    # for image→image we ignore self-match by setting its sim = -inf
    for i in range(N):
        sim_mat[i, i] = -np.inf

    # sort indices descending sim
    ranks = np.argsort(-sim_mat, axis=1)

    aps = {k: [] for k in ks}
    for i in range(N):
        pos = positives[i]
        if not pos:
            continue  # no positives
        ranked = ranks[i]
        # binary relevance at each rank
        rel = np.in1d(ranked, list(pos)).astype(int)
        cum_rel = np.cumsum(rel)
        precisions = cum_rel / (np.arange(len(rel)) + 1)
        for k in ks:
            topk_rel = rel[:k]
            if topk_rel.sum() == 0:
                ap = 0.0
            else:
                # only sum precisions at ranks where rel==1, but cap at k
                ap = (precisions[:k] * topk_rel).sum() / min(len(pos), k)
            aps[k].append(ap)

    return {k: np.mean(aps[k]) for k in ks}


def recall_at_k(sim_mat, ground_truth, ks):
    """
    sim_mat: (M, N) text→image similarity
    ground_truth: list of ints, ground_truth[i] = index of the matching image for text i
    ks: list of cutoffs
    Returns dict {k: Recall@k}
    """
    M = sim_mat.shape[0]
    ranks = np.argsort(-sim_mat, axis=1)
    recalls = {k: 0 for k in ks}
    for i in range(M):
        gt = ground_truth[i]
        ranklist = ranks[i]
        for k in ks:
            if gt in ranklist[:k]:
                recalls[k] += 1
    return {k: recalls[k] / M for k in ks}


def main(args):

    # 1) load labels
    df = pd.read_csv(args.csv_path)
    if "VolumeName" not in df.columns:
        raise ValueError("CSV must contain 'VolumeName'")
    # build per-sample label-tuple
    label_tuples = build_label_tuples(df)

    # map each unique label-tuple to the set of indices having it
    tuple_to_idxs = defaultdict(set)
    for idx, tup in enumerate(label_tuples):
        tuple_to_idxs[tup].add(idx)
    # for each idx, positives = all others with same tup
    positives = [
        tuple_to_idxs[label_tuples[i]] - {i}
        for i in range(len(label_tuples))
    ]

    # 2) load image embeddings
    img_embs, img_ids = load_embeddings(df, args.emb_dir, args.img_emb)
    # 3) image→image retrieval
    sim_ii = cosine_similarity(img_embs)
    mapk = map_at_k(sim_ii, positives, ks=[1, 5, 10, 50])
    print("### Image→Image MAP@K ###")
    for k, v in mapk.items():
        print(f" MAP@{k:2d}: {v:.4f}")

    # 4) load text embeddings
    txt_embs, txt_ids = load_embeddings(df, args.emb_dir, args.txt_emb)
    # ensure same ordering
    assert txt_ids == img_ids, "Image and text sets must be same samples/in same order"

    # 5) text→image retrieval
    sim_ti = cosine_similarity(txt_embs, img_embs)
    # ground truth: each text i matches image i
    gt = list(range(len(txt_ids)))
    recall = recall_at_k(sim_ti, gt, ks=[5, 10, 50, 100])
    print("\n### Text→Image Recall@K ###")
    for k, v in recall.items():
        print(f" Recall@{k:3d}: {v:.4f}")


if __name__ == "__main__":
    
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
