import heapq
import os
import time
import numpy as np


def extract_embeddings_to_numpy(model, dataloader, device, save_prefix=None):
    """Extract L2-normalized embeddings for a split and store them as a NumPy matrix."""
    import torch

    model.eval()
    embs, lbls, idxs = [], [], []

    with torch.no_grad():
        for images, labels, indices in dataloader:
            images = images.to(device)
            out = model(images)
            embs.append(out.cpu().numpy().astype(np.float32))
            lbls.append(labels.numpy().astype(np.int64))
            idxs.append(indices.numpy().astype(np.int64))

    embeddings = np.concatenate(embs, axis=0)
    labels = np.concatenate(lbls, axis=0)
    indices = np.concatenate(idxs, axis=0)

    if save_prefix is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_prefix)), exist_ok=True)
        np.save(f"{save_prefix}_embeddings.npy", embeddings)
        np.save(f"{save_prefix}_labels.npy", labels)
        np.save(f"{save_prefix}_indices.npy", indices)

    return embeddings, labels, indices


def compute_l2_distance_matrix_numpy(queries, gallery):
    """Compute pairwise Euclidean distances between query and gallery matrices."""
    q_sq = np.sum(queries ** 2, axis=1, keepdims=True)
    g_sq = np.sum(gallery ** 2, axis=1, keepdims=True).T
    sq = np.maximum(q_sq - 2.0 * np.dot(queries, gallery.T) + g_sq, 0.0)
    return np.sqrt(sq).astype(np.float32)


def compute_cosine_similarity_matrix_numpy(queries, gallery):
    """Compute pairwise cosine similarities between query and gallery matrices."""
    qn = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    gn = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    return np.dot(qn, gn.T).astype(np.float32)


class BruteForceRetriever:
    """Brute-force L2 and cosine similarity search over a NumPy embedding gallery."""

    def __init__(self, gallery_embeddings, gallery_labels, class_names=None):
        self.gallery_embeddings = np.asarray(gallery_embeddings, dtype=np.float32)
        self.gallery_labels = np.asarray(gallery_labels, dtype=np.int64)
        self.class_names = class_names

    def search_l2(self, query_embeddings, top_k=5, exclude_self_indices=None):
        q = np.asarray(query_embeddings, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]

        dist_mat = compute_l2_distance_matrix_numpy(q, self.gallery_embeddings)
        cos_mat = compute_cosine_similarity_matrix_numpy(q, self.gallery_embeddings)

        if exclude_self_indices is not None:
            for r_i, g_i in enumerate(exclude_self_indices):
                if 0 <= g_i < dist_mat.shape[1]:
                    dist_mat[r_i, g_i] = np.inf

        k = min(top_k, self.gallery_embeddings.shape[0])
        part = np.argpartition(dist_mat, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(q.shape[0])[:, None]
        order = np.argsort(dist_mat[rows, part], axis=1)

        top_idx = part[rows, order]
        top_dist = dist_mat[rows, top_idx]
        top_cos = cos_mat[rows, top_idx]
        top_lbl = self.gallery_labels[top_idx]

        res = {
            "indices": top_idx,
            "distances": top_dist,
            "cosine_similarities": top_cos,
            "labels": top_lbl,
        }
        if self.class_names is not None:
            res["class_names"] = [[self.class_names[int(x)] for x in row] for row in top_lbl]
        return res

    def search_cosine(self, query_embeddings, top_k=5, exclude_self_indices=None):
        q = np.asarray(query_embeddings, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]

        cos_mat = compute_cosine_similarity_matrix_numpy(q, self.gallery_embeddings)
        dist_mat = compute_l2_distance_matrix_numpy(q, self.gallery_embeddings)

        if exclude_self_indices is not None:
            for r_i, g_i in enumerate(exclude_self_indices):
                if 0 <= g_i < cos_mat.shape[1]:
                    cos_mat[r_i, g_i] = -np.inf

        k = min(top_k, self.gallery_embeddings.shape[0])
        part = np.argpartition(-cos_mat, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(q.shape[0])[:, None]
        order = np.argsort(-cos_mat[rows, part], axis=1)

        top_idx = part[rows, order]
        top_cos = cos_mat[rows, top_idx]
        top_dist = dist_mat[rows, top_idx]
        top_lbl = self.gallery_labels[top_idx]

        res = {
            "indices": top_idx,
            "distances": top_dist,
            "cosine_similarities": top_cos,
            "labels": top_lbl,
        }
        if self.class_names is not None:
            res["class_names"] = [[self.class_names[int(x)] for x in row] for row in top_lbl]
        return res


class _KDNode:
    __slots__ = ("point", "index", "label", "axis", "left", "right")

    def __init__(self, point, index, label, axis, left=None, right=None):
        self.point = point
        self.index = index
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right


class NumPyKDTree:
    """KD-Tree index built from scratch in NumPy for exact nearest neighbor search."""

    def __init__(self, embeddings, labels):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.dim = self.embeddings.shape[1]
        self.root = self._build(np.arange(len(self.embeddings)), 0)

    def _build(self, indices, depth):
        if len(indices) == 0:
            return None
        axis = depth % self.dim
        coords = self.embeddings[indices, axis]
        mid = len(indices) // 2
        order = np.argpartition(coords, mid)
        indices = indices[order]
        med_idx = int(indices[mid])

        return _KDNode(
            point=self.embeddings[med_idx],
            index=med_idx,
            label=int(self.labels[med_idx]),
            axis=axis,
            left=self._build(indices[:mid], depth + 1),
            right=self._build(indices[mid + 1:], depth + 1),
        )

    def query(self, query_point, top_k=5, exclude_index=None):
        q = np.asarray(query_point, dtype=np.float32).ravel()
        heap = []

        def _search(node):
            if node is None:
                return
            if exclude_index is None or node.index != exclude_index:
                diff = q - node.point
                sq_d = float(np.dot(diff, diff))
                if len(heap) < top_k:
                    heapq.heappush(heap, (-sq_d, node.index, node.label))
                elif sq_d < -heap[0][0]:
                    heapq.heapreplace(heap, (-sq_d, node.index, node.label))

            axis = node.axis
            delta = float(q[axis] - node.point[axis])
            near = node.left if delta <= 0 else node.right
            far = node.right if delta <= 0 else node.left

            _search(near)
            if len(heap) < top_k or (delta * delta) < -heap[0][0]:
                _search(far)

        _search(self.root)
        items = sorted(heap, key=lambda x: -x[0])
        dists = np.array([np.sqrt(max(0.0, -x[0])) for x in items], dtype=np.float32)
        idxs = np.array([x[1] for x in items], dtype=np.int64)
        lbls = np.array([x[2] for x in items], dtype=np.int64)
        return idxs, dists, lbls


def k_reciprocal_rerank(query_embeddings, gallery_embeddings, k1=20, k2=6, lambda_value=0.3):
    """Re-rank gallery distances using k-reciprocal nearest neighbors in NumPy."""
    all_feats = np.concatenate([query_embeddings, gallery_embeddings], axis=0)
    n_q = query_embeddings.shape[0]
    n_all = all_feats.shape[0]

    orig_dist = compute_l2_distance_matrix_numpy(all_feats, all_feats) ** 2
    orig_dist = orig_dist / np.maximum(np.max(orig_dist, axis=1, keepdims=True), 1e-12)

    rank = np.argsort(orig_dist, axis=1)
    V = np.zeros((n_all, n_all), dtype=np.float32)

    for i in range(n_all):
        fwd = rank[i, :k1 + 1]
        bwd = rank[fwd, :k1 + 1]
        fi = np.where(bwd == i)[0]
        k_recip = fwd[fi]
        expanded = k_recip.copy()

        for cand in k_recip:
            ck = max(1, int(round(k1 / 2)))
            cfwd = rank[cand, :ck + 1]
            cbwd = rank[cfwd, :ck + 1]
            cfi = np.where(cbwd == cand)[0]
            crecip = cfwd[cfi]
            if len(np.intersect1d(crecip, k_recip)) > (2.0 / 3.0) * len(crecip):
                expanded = np.union1d(expanded, crecip)

        w = np.exp(-orig_dist[i, expanded])
        V[i, expanded] = w / np.maximum(np.sum(w), 1e-12)

    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(n_all):
            V_qe[i, :] = np.mean(V[rank[i, :k2], :], axis=0)
        V = V_qe

    q_V = V[:n_q]
    g_V = V[n_q:]
    jaccard = np.zeros((n_q, n_all - n_q), dtype=np.float32)
    for i in range(n_q):
        mn = np.sum(np.minimum(q_V[i:i + 1, :], g_V), axis=1)
        mx = np.sum(np.maximum(q_V[i:i + 1, :], g_V), axis=1)
        jaccard[i] = 1.0 - mn / np.maximum(mx, 1e-12)

    return (1.0 - lambda_value) * jaccard + lambda_value * orig_dist[:n_q, n_q:]


def benchmark_search_latency(query_embeddings, gallery_embeddings, top_k=5, num_runs=10):
    """Compare query latency of NumPy search against FAISS."""
    q = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    g = np.ascontiguousarray(gallery_embeddings, dtype=np.float32)

    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = compute_l2_distance_matrix_numpy(q, g)
    np_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    out = {"numpy_brute_force_ms": np_ms, "numpy_ms_per_query": np_ms / max(1, len(q))}

    try:
        import faiss

        dim = g.shape[1]
        flat = faiss.IndexFlatL2(dim)
        flat.add(g)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            flat.search(q, top_k)
        out["faiss_cpu_flat_ms"] = ((time.perf_counter() - t0) / num_runs) * 1000.0
    except ImportError:
        pass

    return out
