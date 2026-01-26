import os
import argparse
from typing import List, Tuple

import umap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoConfig

sns.set(style="whitegrid", font_scale=1.1)


def load_texts_for_section(df: pd.DataFrame, section_prefix: str) -> Tuple[List[str], List[str], List[str], List[int]]:
    """
    Returns lists:
    - originals: original texts
    - rew1: rewrite 1 texts
    - rew2: rewrite 2 texts
    - indices: original dataframe indices corresponding to returned lists
    Only returns rows where original is non-empty.
    """
    originals = []
    rew1 = []
    rew2 = []
    idxs = []

    for i, row in df.iterrows():
        orig = row.get(f"{section_prefix}_EN", None)
        r1 = row.get(f"{section_prefix}_1", None)
        r2 = row.get(f"{section_prefix}_2", None)
        # Keep only if original exists (you can adjust filtering here)
        if isinstance(orig, str) and orig.strip():
            originals.append(orig.strip())
            rew1.append(r1.strip() if isinstance(r1, str) else "")
            rew2.append(r2.strip() if isinstance(r2, str) else "")
            idxs.append(i)
    return originals, rew1, rew2, idxs


def embed_texts_radbert(
    texts,
    tokenizer,
    model,
    device="cpu",
    batch_size=16,
    max_length=512,
):
    """
    Embeds texts using RadBERT (RoBERTa) with attention-masked mean pooling.
    Returns L2-normalized embeddings suitable for cosine similarity.
    """

    model.eval()
    model.to(device)

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            # IMPORTANT: call the underlying encoder, not the classifier
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            last_hidden = outputs.last_hidden_state  # (B, C, H)

            # Attention-masked mean pooling
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            emb = summed / counts  # (B, H)

        # Normalize for cosine similarity
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


def retrieval_r_at_k(similarities: np.ndarray, target_indices: np.ndarray, k: int = 1) -> float:
    """
    similarities: (Q, N) similarities between Q queries and N candidate originals
    target_indices: (Q,) integer indices of the correct original in 0..N-1
    returns recall@k fraction
    """
    Q, N = similarities.shape
    ranks = np.argsort(-similarities, axis=1)  # descending
    topk = ranks[:, :k]
    hits = [(target_indices[q] in topk[q]) for q in range(Q)]
    return float(np.mean(hits))


def per_sample_metrics(orig_emb, r1_emb, r2_emb):
    """
    Compute per-sample pairwise similarities and some summary stats.
    Returns DataFrame with columns:
      sim_orig_r1, sim_orig_r2, sim_r1_r2
    """
    sims_orig_r1 = cosine_similarity(orig_emb, r1_emb).diagonal()
    sims_orig_r2 = cosine_similarity(orig_emb, r2_emb).diagonal()
    sims_r1_r2 = cosine_similarity(r1_emb, r2_emb).diagonal()
    df = pd.DataFrame({
        "sim_orig_r1": sims_orig_r1,
        "sim_orig_r2": sims_orig_r2,
        "sim_r1_r2": sims_r1_r2
    })
    return df


def evaluate_section(
    df: pd.DataFrame,
    section_prefix: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    outdir: str,
    device: str = "cpu",
    subset_for_umap: int = 80,
    seed: int = 42
):
    """Evaluate a section and return data for combined plots"""
    originals, rew1, rew2, idxs = load_texts_for_section(df, section_prefix)
    originals, rew1, rew2, idxs = originals[:150], rew1[:150], rew2[:150], idxs[:150]
    n = len(originals)
    print(f"[{section_prefix}] n = {n} samples (originals present)")

    # Combine lists for embedding efficiency
    all_texts = originals + rew1 + rew2
    emb_all = embed_texts_radbert(
        all_texts,
        tokenizer,
        model,
        device=device,
        batch_size=16,
    )
    orig_emb = emb_all[0:n]
    r1_emb = emb_all[n:2*n]
    r2_emb = emb_all[2*n:3*n]

    # pairwise per-sample metrics
    per_sample_df = per_sample_metrics(orig_emb, r1_emb, r2_emb)

    # baseline: per-original similarity distribution to other originals (exclude self)
    orig_sim_matrix = cosine_similarity(orig_emb, orig_emb)
    orig_sim_to_others = []
    for i in range(n):
        sims = np.delete(orig_sim_matrix[i], i)  # remove self
        orig_sim_to_others.append(sims)
    # flattened baseline distribution for plotting
    baseline_flat = np.concatenate(orig_sim_to_others).ravel()

    # Retrieval evaluation: treat each rewrite as query to set of originals
    sim_r1_to_orig = cosine_similarity(r1_emb, orig_emb)  # (n, n)
    sim_r2_to_orig = cosine_similarity(r2_emb, orig_emb)
    # targets are index 0..n-1 in same order
    target_idxs = np.arange(n, dtype=int)
    r1_r_at_1 = retrieval_r_at_k(sim_r1_to_orig, target_idxs, k=1)
    r1_r_at_5 = retrieval_r_at_k(sim_r1_to_orig, target_idxs, k=5)
    r2_r_at_1 = retrieval_r_at_k(sim_r2_to_orig, target_idxs, k=1)
    r2_r_at_5 = retrieval_r_at_k(sim_r2_to_orig, target_idxs, k=5)

    # percentile ranks: for each rewrite, compute percentile rank of paired original similarity among similarities to all originals
    def percentile_ranks(sim_q_to_orig):
        # sim_q_to_orig: (n, n) sims of each query to each original
        ranks = []
        for i in range(n):
            sim_pair = sim_q_to_orig[i, i]
            sims = sim_q_to_orig[i]
            # percentile: fraction of sims <= sim_pair
            pct = (np.sum(sims <= sim_pair) / len(sims)) * 100.0
            ranks.append(pct)
        return np.array(ranks)

    r1_percentiles = percentile_ranks(sim_r1_to_orig)
    r2_percentiles = percentile_ranks(sim_r2_to_orig)
    per_sample_df["r1_percentile_of_pair"] = r1_percentiles
    per_sample_df["r2_percentile_of_pair"] = r2_percentiles

    # add mean baseline sim for each original
    per_sample_df["orig_mean_sim_to_others"] = np.array([arr.mean() for arr in orig_sim_to_others])
    per_sample_df["orig_median_sim_to_others"] = np.array([np.median(arr) for arr in orig_sim_to_others])

    # Save per-sample CSV
    per_sample_df["index"] = idxs
    per_sample_df.to_csv(os.path.join(outdir, f"{section_prefix}_per_sample_similarity.csv"), index=False)

    # Print summary
    def summary_values(arr):
        return f"{arr.mean():.4f} ± {arr.std():.4f} (median {np.median(arr):.4f})"

    print(f"Summary for {section_prefix}:")
    print(f"  mean(sim orig-r1): {summary_values(per_sample_df['sim_orig_r1'].values)}")
    print(f"  mean(sim orig-r2): {summary_values(per_sample_df['sim_orig_r2'].values)}")
    print(f"  mean(sim r1-r2)  : {summary_values(per_sample_df['sim_r1_r2'].values)}")
    print(f"  mean(orig sim to other originals): {summary_values(per_sample_df['orig_mean_sim_to_others'].values)}")
    print(f"  Retrieval R@1 (r1->orig): {r1_r_at_1:.4f}, R@5: {r1_r_at_5:.4f}")
    print(f"  Retrieval R@1 (r2->orig): {r2_r_at_1:.4f}, R@5: {r2_r_at_5:.4f}")
    print("  Saved per-sample CSV.")

    # UMAP visualization for subset
    rng = np.random.RandomState(seed)
    subset_n = min(subset_for_umap, n)
    chosen = rng.choice(n, size=subset_n, replace=False)
    emb_subset = []
    labels = []
    sample_ids = []
    for i, idx in enumerate(chosen):
        emb_subset.append(orig_emb[idx])
        labels.append("orig")
        sample_ids.append(i)
        emb_subset.append(r1_emb[idx])
        labels.append("rew1")
        sample_ids.append(i)
        emb_subset.append(r2_emb[idx])
        labels.append("rew2")
        sample_ids.append(i)
    emb_subset = np.vstack(emb_subset)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)
    umap_2d = reducer.fit_transform(emb_subset)

    # Plot UMAP: color by sample id, shape by version
    plt.figure(figsize=(10, 8))
    unique_samples = sorted(set(sample_ids))
    palette = sns.color_palette("husl", n_colors=len(unique_samples))
    marker_map = {"orig": "o", "rew1": "s", "rew2": "X"}

    for i, sid in enumerate(unique_samples):
        # pick points for sample i
        idxs_sample = [j for j, s in enumerate(sample_ids) if s == sid]
        xs = umap_2d[idxs_sample, 0]
        ys = umap_2d[idxs_sample, 1]
        vers = [labels[j] for j in idxs_sample]
        for xi, yi, v in zip(xs, ys, vers):
            plt.scatter(xi, yi, marker=marker_map[v], color=palette[i], s=60, alpha=0.9)
        # connect points with a line
        plt.plot(xs, ys, linestyle='-', color=palette[i], linewidth=0.8, alpha=0.6)

    plt.title(f"{section_prefix} UMAP (subset={subset_n} samples). Lines connect original->rewrites")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    # create legend for markers
    for v, m in marker_map.items():
        plt.scatter([], [], marker=m, color="k", label=v)
    plt.legend(frameon=True, loc="best")
    plt.savefig(os.path.join(outdir, f"{section_prefix}_umap_subset{subset_n}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Return the per-sample dataframe and summary metrics plus data for combined plots
    summary = {
        "r1_r_at_1": r1_r_at_1, "r1_r_at_5": r1_r_at_5,
        "r2_r_at_1": r2_r_at_1, "r2_r_at_5": r2_r_at_5,
        "mean_sim_orig_r1": per_sample_df["sim_orig_r1"].mean(),
        "mean_sim_orig_r2": per_sample_df["sim_orig_r2"].mean()
    }
    
    # Return data for combined plots
    return {
        "per_sample_df": per_sample_df,
        "summary": summary,
        "rewrites_to_orig": np.concatenate([per_sample_df["sim_orig_r1"].values, per_sample_df["sim_orig_r2"].values]),
        "orig_to_others": baseline_flat,
        "percentiles": np.concatenate([r1_percentiles, r2_percentiles])
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, required=True, help="Excel file with report sections")
    parser.add_argument("--model_name", type=str, default="zzxslp/RadBERT-RoBERTa-4m", help="Transformers model name")
    parser.add_argument("--outdir", type=str, default="results", help="where to write outputs")
    parser.add_argument("--device", type=str, default="cpu", help="device for model (cpu or cuda)")
    parser.add_argument("--subset_for_umap", type=int, default=80, help="number of samples to visualize with UMAP")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_excel(args.xlsx)
    print(f"Loaded Excel with {len(df)} rows")

    print("Loading model:", args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Evaluate both sections if present (adjust names to match your CSV)
    sections = []
    if any(col.startswith("Impressions") for col in df.columns):
        sections.append("Impressions")
    if any(col.startswith("Findings") for col in df.columns):
        sections.append("Findings")

    if not sections:
        raise RuntimeError("CSV doesn't contain 'Impressions' or 'Findings' columns matching the expected naming scheme.")

    # Evaluate each section and collect results
    overall_summary = {}
    section_results = {}
    
    for sec in sections:
        sec_outdir = os.path.join(args.outdir, sec)
        os.makedirs(sec_outdir, exist_ok=True)
        
        result = evaluate_section(
            df,
            sec,
            tokenizer,
            model,
            outdir=sec_outdir,
            device=args.device,
            subset_for_umap=args.subset_for_umap,
            seed=args.seed
        )
        overall_summary[sec] = result["summary"]
        section_results[sec] = result

    # Create combined violin plot
    plot_data = []
    for sec in sections:
        result = section_results[sec]
        # Add rewrite similarities
        for sim in result["rewrites_to_orig"]:
            plot_data.append({"section": sec, "type": "Rewrite", "similarity": sim})
        # Add original-to-other similarities
        for sim in result["orig_to_others"]:
            plot_data.append({"section": sec, "type": "Other originals", "similarity": sim})
    
    plot_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x="section", y="similarity", hue="type", data=plot_df, inner="quartile", cut=0, split=True)
    ax.set_title("Within-pair vs across-original similarities")
    ax.set_ylabel("Cosine Similarity")
    ax.set_xlabel("Section")
    plt.legend(title="Comparison", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "combined_similarity_violin.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Create combined CDF plot
    plt.figure(figsize=(8, 5))
    for sec in sections:
        result = section_results[sec]
        sns.ecdfplot(result["percentiles"], label=sec)
    plt.xlabel("Percentile rank of paired original among all originals")
    plt.ylabel("ECDF")
    plt.title("Percentile rank of paired original (higher = better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "combined_percentile_cdf.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Save overall summary
    summary_df = pd.DataFrame.from_dict(overall_summary, orient='index')
    summary_df.to_csv(os.path.join(args.outdir, "summary_metrics.csv"))
    print("Done. Results written to", args.outdir)


if __name__ == "__main__":
    main()
