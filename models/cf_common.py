"""Sparse interaction matrix, exact blockwise cosine neighbors, and fallback helpers."""
from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import polars as pl
from scipy import sparse

RECOMMENDATION_SCHEMA = {
    "user_id": pl.Int64, "product_id": pl.Int64, "rank": pl.Int64,
    "score": pl.Float64, "source": pl.String,
    "reason_product_id": pl.Int64, "reason_user_id": pl.Int64,
    "reason_similarity": pl.Float64,
}


class SparseInteractions:
    """Only historical interaction_score is used; no target label is accepted."""
    def __init__(self, interactions: pl.DataFrame):
        needed = {"user_id", "product_id", "interaction_score"}
        if not needed <= set(interactions.columns) or not interactions.height:
            raise ValueError("Need a nonempty historical user-product interaction table")
        if interactions.select("user_id", "product_id").unique().height != interactions.height:
            raise ValueError("Duplicate user-product pairs")
        if interactions.select(list(needed)).null_count().sum_horizontal().item():
            raise ValueError("Null matrix identifiers or scores")
        self.user_ids = interactions["user_id"].unique().sort().to_numpy().astype(np.int64)
        self.product_ids = interactions["product_id"].unique().sort().to_numpy().astype(np.int64)
        self.user_index = {int(user): pos for pos, user in enumerate(self.user_ids)}
        self.product_index = {int(product): pos for pos, product in enumerate(self.product_ids)}
        users = np.searchsorted(self.user_ids, interactions["user_id"].to_numpy())
        products = np.searchsorted(self.product_ids, interactions["product_id"].to_numpy())
        scores = interactions["interaction_score"].to_numpy().astype(np.float32)
        if not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise ValueError("Interaction scores must be finite and non-negative")
        self.matrix = sparse.csr_matrix(
            (scores, (users, products)), shape=(len(self.user_ids), len(self.product_ids))
        )
        self.matrix.eliminate_zeros()
        self.matrix.sort_indices()


def normalize_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    squared_norms = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    inverse = np.zeros_like(squared_norms, dtype=np.float32)
    valid = squared_norms > 0
    inverse[valid] = 1 / np.sqrt(squared_norms[valid])
    return matrix.multiply(inverse[:, None]).tocsr()


def select_top(indices: np.ndarray, scores: np.ndarray, k: int,
               exclude_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rank by descending score then ascending (sorted ID) index, even at ties."""
    if k <= 0:
        raise ValueError("k must be positive")
    mask = np.isfinite(scores) & (scores > 0)
    if exclude_index is not None:
        mask &= indices != exclude_index
    indices, scores = indices[mask], scores[mask]
    if len(indices) > k:
        cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
        above = np.flatnonzero(scores > cutoff)
        tied = np.flatnonzero(scores == cutoff)
        tied = tied[np.argsort(indices[tied], kind="stable")][:k - len(above)]
        chosen = np.concatenate([above, tied])
        indices, scores = indices[chosen], scores[chosen]
    order = np.lexsort((indices, -scores))
    return indices[order].astype(np.int32, copy=False), scores[order].astype(np.float32, copy=False)


def cosine_neighbors(vectors: sparse.csr_matrix, query_rows: np.ndarray, neighbors: int,
                     block_size: int = 64,
                     progress: Callable[[int, int], None] | None = None) -> sparse.csr_matrix:
    """Exact positive cosine neighbors; only block x entity intermediates exist.

    Rows in the returned CSR retain similarity-rank order, not column order.
    Self-neighbors are excluded. Zero cosine never becomes a neighbor.
    """
    if neighbors <= 0 or block_size <= 0:
        raise ValueError("neighbors and block_size must be positive")
    vectors_t = vectors.transpose().tocsc()
    row_indices, row_scores, indptr = [], [], [0]
    for offset in range(0, len(query_rows), block_size):
        chunk = query_rows[offset:offset + block_size]
        similarities = (vectors[chunk] @ vectors_t).tocsr()
        similarities.sum_duplicates()
        for local, source in enumerate(chunk):
            start, end = similarities.indptr[local:local + 2]
            ids, values = select_top(similarities.indices[start:end],
                                     similarities.data[start:end], neighbors, int(source))
            row_indices.append(ids)
            row_scores.append(values)
            indptr.append(indptr[-1] + len(ids))
        if progress is not None:
            progress(min(offset + block_size, len(query_rows)), len(query_rows))
    indices = np.concatenate(row_indices) if row_indices else np.array([], dtype=np.int32)
    scores = np.concatenate(row_scores) if row_scores else np.array([], dtype=np.float32)
    return sparse.csr_matrix((scores, indices, np.array(indptr, dtype=np.int64)),
                             shape=(len(query_rows), vectors.shape[0]))


def adjusted_item_neighbors(
    vectors: sparse.csr_matrix,
    binary_vectors: sparse.csr_matrix,
    query_rows: np.ndarray,
    neighbors: int,
    shrinkage: float = 0.0,
    min_common_users: int = 1,
    item_popularity: np.ndarray | None = None,
    popularity_alpha: float = 0.0,
    block_size: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> sparse.csr_matrix:
    """Item neighbors with significance weighting and optional popularity penalty.

    `vectors` contains the weighted item vectors used for cosine similarity;
    `binary_vectors` is only used to count distinct common users. Both have
    shape (items, users). The returned rows are similarity-ranked and exclude
    self-neighbors.
    """
    if neighbors <= 0 or block_size <= 0:
        raise ValueError("neighbors and block_size must be positive")
    if (not np.isfinite(shrinkage) or not np.isfinite(popularity_alpha)
            or shrinkage < 0 or min_common_users < 1 or popularity_alpha < 0):
        raise ValueError("shrinkage/alpha must be non-negative and min support >= 1")
    if vectors.shape != binary_vectors.shape:
        raise ValueError("Weighted and binary item matrices must have the same shape")
    if popularity_alpha and item_popularity is None:
        raise ValueError("item_popularity is required when popularity_alpha > 0")
    if item_popularity is not None and len(item_popularity) != vectors.shape[0]:
        raise ValueError("item_popularity length must equal item count")

    vector_t = vectors.transpose().tocsr()
    use_support = min_common_users > 1 or shrinkage > 0
    binary_t = binary_vectors.transpose().tocsr() if use_support else None
    row_indices, row_scores, indptr = [], [], [0]
    for offset in range(0, len(query_rows), block_size):
        chunk = query_rows[offset:offset + block_size]
        similarities = (vectors[chunk] @ vector_t).tocsr()
        similarities.sum_duplicates()
        if binary_t is not None and similarities.nnz:
            supports = (binary_vectors[chunk] @ binary_t).tocsr()
            supports.sum_duplicates()
            if (np.array_equal(similarities.indptr, supports.indptr)
                    and np.array_equal(similarities.indices, supports.indices)):
                counts = supports.data
            else:
                rows = np.repeat(np.arange(len(chunk)), np.diff(similarities.indptr))
                counts = np.asarray(supports[rows, similarities.indices]).ravel()
            if shrinkage:
                similarities.data *= counts / (counts + shrinkage)
            if min_common_users > 1:
                similarities.data[counts < min_common_users] = 0
                similarities.eliminate_zeros()
        if popularity_alpha and similarities.nnz:
            target_popularity = np.maximum(item_popularity[similarities.indices], 1.0)
            similarities.data *= np.power(target_popularity, -popularity_alpha)
        for local, source in enumerate(chunk):
            start, end = similarities.indptr[local:local + 2]
            ids, values = select_top(similarities.indices[start:end],
                                     similarities.data[start:end], neighbors, int(source))
            row_indices.append(ids)
            row_scores.append(values)
            indptr.append(indptr[-1] + len(ids))
        if progress is not None:
            progress(min(offset + block_size, len(query_rows)), len(query_rows))
    indices = np.concatenate(row_indices) if row_indices else np.array([], dtype=np.int32)
    scores = np.concatenate(row_scores) if row_scores else np.array([], dtype=np.float32)
    return sparse.csr_matrix((scores, indices, np.array(indptr, dtype=np.int64)),
                             shape=(len(query_rows), vectors.shape[0]))


def truncate_neighbors(graph: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Keep the first k SIMILARITY-RANKED entries per row, not k column IDs."""
    if k <= 0:
        raise ValueError("k must be positive")
    indices, values, pointers = [], [], [0]
    for row in range(graph.shape[0]):
        start = graph.indptr[row]
        end = min(start + k, graph.indptr[row + 1])
        indices.append(graph.indices[start:end])
        values.append(graph.data[start:end])
        pointers.append(pointers[-1] + end - start)
    return sparse.csr_matrix((np.concatenate(values) if values else [],
                              np.concatenate(indices) if indices else [],
                              np.asarray(pointers, dtype=np.int64)), shape=graph.shape)


def ranked_candidates(scores: sparse.csr_matrix, product_ids: np.ndarray,
                      blocked: set[int], k: int) -> list[tuple[int, float]]:
    scores = scores.tocsr()
    scores.sum_duplicates()
    allowed = np.fromiter((int(product_ids[pos]) not in blocked for pos in scores.indices),
                          dtype=bool, count=len(scores.indices))
    indices, values = select_top(scores.indices[allowed], scores.data[allowed], k)
    return [(int(product_ids[index]), float(value)) for index, value in zip(indices, values)]


def make_recommendations(users: Iterable[int], k: int, blocked: Mapping[int, set[int]],
                         fallback_ranking: pl.DataFrame, source: str,
                         score_user: Callable[[int, set[int], int], list[dict]]) -> pl.DataFrame:
    """CF ranks first; popularity fills empty slots. Scores across sources do not mix."""
    if k <= 0:
        raise ValueError("k must be positive")
    fallback = [(int(p), float(s)) for p, s in fallback_ranking.select("product_id", "score").iter_rows()]
    rows = []
    for user in sorted(set(users)):
        excluded = blocked.get(user, set())
        items = score_user(user, excluded, k)
        selected = set()
        for item in items:
            product = item["product_id"]
            if product in excluded or product in selected:
                raise ValueError("CF emitted duplicate or excluded item")
            selected.add(product)
            rows.append({"user_id": user, "rank": len(selected), "source": source, **item})
        if len(selected) < k:
            for product, score in fallback:
                if product in excluded or product in selected:
                    continue
                selected.add(product)
                rows.append({"user_id": user, "product_id": product, "rank": len(selected),
                             "score": score, "source": "popularity_fallback",
                             "reason_product_id": None, "reason_user_id": None,
                             "reason_similarity": None})
                if len(selected) == k:
                    break
    return pl.DataFrame(rows, schema=RECOMMENDATION_SCHEMA)
