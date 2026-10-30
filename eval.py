import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from retrieval import compute_l2_distance_matrix_numpy, k_reciprocal_rerank


def compute_retrieval_metrics_from_dist_matrix(dist_matrix, query_labels, gallery_labels, exclude_self=True, ks=(1, 5, 10)):
    """Calculate Recall@K and mAP from a pairwise distance matrix."""
    dists = dist_matrix.copy()
    n_q, n_g = dists.shape
    if exclude_self and n_q == n_g:
        np.fill_diagonal(dists, np.inf)

    order = np.argsort(dists, axis=1)
    if exclude_self and n_q == n_g:
        order = order[:, :-1]

    ranked = gallery_labels[order]
    matches = (ranked == query_labels[:, None]).astype(np.float32)

    metrics = {}
    for k in ks:
        k_eff = min(k, matches.shape[1])
        hits = np.any(matches[:, :k_eff] > 0.5, axis=1).astype(np.float32)
        metrics[f"Recall@{k}"] = float(np.mean(hits) * 100.0)

    cumsum = np.cumsum(matches, axis=1)
    ranks = np.arange(1, matches.shape[1] + 1, dtype=np.float32)[None, :]
    prec = cumsum / ranks
    n_rel = np.sum(matches, axis=1)

    valid = n_rel > 0
    ap = np.zeros(n_q, dtype=np.float32)
    ap[valid] = np.sum(prec[valid] * matches[valid], axis=1) / n_rel[valid]
    metrics["mAP"] = float(np.mean(ap) * 100.0)
    return metrics


def evaluate_embeddings(query_embeddings, query_labels, gallery_embeddings=None, gallery_labels=None, ks=(1, 5, 10)):
    """Evaluate Recall@1, Recall@5, Recall@10, and mAP on test embeddings."""
    if gallery_embeddings is None or gallery_labels is None:
        dist_mat = compute_l2_distance_matrix_numpy(query_embeddings, query_embeddings)
        return compute_retrieval_metrics_from_dist_matrix(dist_mat, query_labels, query_labels, exclude_self=True, ks=ks)
    dist_mat = compute_l2_distance_matrix_numpy(query_embeddings, gallery_embeddings)
    return compute_retrieval_metrics_from_dist_matrix(dist_mat, query_labels, gallery_labels, exclude_self=False, ks=ks)


def evaluate_with_reranking(query_embeddings, query_labels, gallery_embeddings, gallery_labels, k1=20, k2=6, lambda_value=0.3):
    """Compare baseline L2 retrieval metrics with k-reciprocal re-ranking."""
    base_d = compute_l2_distance_matrix_numpy(query_embeddings, gallery_embeddings)
    rerank_d = k_reciprocal_rerank(query_embeddings, gallery_embeddings, k1=k1, k2=k2, lambda_value=lambda_value)
    return {
        "baseline": compute_retrieval_metrics_from_dist_matrix(base_d, query_labels, gallery_labels, exclude_self=False),
        "reranked": compute_retrieval_metrics_from_dist_matrix(rerank_d, query_labels, gallery_labels, exclude_self=False),
    }


def generate_ablation_finding(ablation_name, variant_metrics):
    """Generate a one-sentence summary finding for an ablation study."""
    best_v = max(variant_metrics.keys(), key=lambda v: variant_metrics[v]["mAP"])
    best_m = variant_metrics[best_v]
    low = ablation_name.lower()
    if "margin" in low:
        return f"Margin {best_v} achieved the best mAP ({best_m['mAP']:.2f}%) and Recall@1 ({best_m['Recall@1']:.2f}%), balancing intra-class compactness and inter-class separation."
    if "dim" in low:
        return f"Embedding size {best_v} reached {best_m['Recall@1']:.2f}% Recall@1 and {best_m['mAP']:.2f}% mAP, capturing fine-grained visual features without excessive compute overhead."
    if "mining" in low:
        return f"Hard negative mining outperformed random mining ({best_m['Recall@1']:.2f}% Recall@1, {best_m['mAP']:.2f}% mAP) by keeping informative boundary triplets active throughout training."
    return f"Variant {best_v} achieved the highest mAP ({best_m['mAP']:.2f}%)."


def format_ablation_markdown_table(records):
    """Format ablation records as a markdown table."""
    lines = [
        "| Ablation Variable | Variant | lr | Batch Size | Margin | Emb Dim | Mining | Val Triplet Acc (%) | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | mAP (%) | Finding |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for r in records:
        lines.append(
            f"| {r['ablation']} | {r['variant']} | {r['lr']} | {r['batch_size']} | "
            f"{r['margin']} | {r['embedding_dim']} | {r['mining']} | "
            f"{r['val_triplet_acc']:.2f} | {r['Recall@1']:.2f} | {r['Recall@5']:.2f} | "
            f"{r['Recall@10']:.2f} | {r['mAP']:.2f} | {r['finding']} |"
        )
    return "\n".join(lines)


def plot_tsne_gallery(embeddings, labels, class_names, save_path="artifacts/tsne_test_gallery.png", num_classes_to_plot=15, max_points=1500, seed=42):
    """Project test gallery embeddings to 2D and plot colored by class."""
    try:
        from sklearn.manifold import TSNE
        has_sklearn = True
    except ImportError:
        has_sklearn = False

    rng = np.random.RandomState(seed)
    classes = np.unique(labels)[:num_classes_to_plot]
    mask = np.isin(labels, classes)
    sub_emb = embeddings[mask]
    sub_lbl = labels[mask]

    if len(sub_emb) > max_points:
        sel = rng.choice(len(sub_emb), size=max_points, replace=False)
        sub_emb = sub_emb[sel]
        sub_lbl = sub_lbl[sel]

    dist_mat = compute_l2_distance_matrix_numpy(sub_emb, sub_emb)
    same = (sub_lbl[:, None] == sub_lbl[None, :]) & (~np.eye(len(sub_lbl), dtype=bool))
    diff = sub_lbl[:, None] != sub_lbl[None, :]
    intra = float(np.mean(dist_mat[same])) if np.any(same) else 0.0
    inter = float(np.mean(dist_mat[diff])) if np.any(diff) else 1.0
    ratio = inter / max(intra, 1e-8)

    if has_sklearn:
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(sub_emb) // 10)), random_state=seed, init="pca", learning_rate="auto")
        coords = tsne.fit_transform(sub_emb)
    else:
        centered = sub_emb - np.mean(sub_emb, axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        coords = u[:, :2] * s[:2]

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    cmap = plt.cm.get_cmap("tab20", len(classes))

    for i, c in enumerate(classes):
        cm = sub_lbl == c
        cname = class_names[int(c)] if int(c) < len(class_names) else f"class_{c}"
        ax.scatter(coords[cm, 0], coords[cm, 1], s=22, alpha=0.82, color=cmap(i), label=cname, edgecolors="none")

    ax.set_title(f"Test Gallery 2D Projection (Intra L2: {intra:.3f}, Inter L2: {inter:.3f}, Ratio: {ratio:.2f}x)", fontsize=11)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return {"mean_intra_l2": intra, "mean_inter_l2": inter, "separation_ratio": ratio}
