"""Weighted cosine Item-CF with blockwise sparse item-neighbor graph."""
from __future__ import annotations
from numbers import Integral

import numpy as np
from .cf_common import (SparseInteractions, normalize_rows, cosine_neighbors, adjusted_item_neighbors,
                        truncate_neighbors, ranked_candidates, make_recommendations)


class ItemCF:
    def __init__(self, data: SparseInteractions, max_neighbors=100, block_size=64,
                 *, shrinkage=0.0, min_common_users=1, iuf=False, popularity_alpha=0.0):
        for name, value in [("max_neighbors", max_neighbors), ("block_size", block_size),
                            ("min_common_users", min_common_users)]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(iuf, (bool, np.bool_)):
            raise ValueError("iuf must be boolean")
        self.data = data
        self.max_neighbors = max_neighbors
        self.block_size = block_size
        self.shrinkage = float(shrinkage)
        self.min_common_users = int(min_common_users)
        self.iuf = bool(iuf)
        self.popularity_alpha = float(popularity_alpha)
        if not all(np.isfinite(value) and value >= 0 for value in (self.shrinkage, self.popularity_alpha)):
            raise ValueError("shrinkage and popularity_alpha must be finite and non-negative")
        self.graph = None

    def fit(self, progress=None):
        weighted = self.data.matrix.tocsr()
        if self.iuf:
            user_item_count = np.diff(weighted.indptr).astype(np.float32)
            # Cosine sees each weight twice, so scale by sqrt(IUF).
            scale = 1.0 / np.sqrt(np.log1p(np.maximum(user_item_count, 1.0)))
            weighted = weighted.multiply(scale[:, None]).tocsr()
        item_vectors = normalize_rows(weighted.transpose().tocsr())
        query_rows = np.arange(item_vectors.shape[0])
        if not (self.shrinkage or self.min_common_users > 1 or self.popularity_alpha):
            self.graph = cosine_neighbors(item_vectors, query_rows, self.max_neighbors,
                                          self.block_size, progress)
            return self

        binary = self.data.matrix.copy()
        binary.data = np.ones(binary.nnz, dtype=np.int32)
        binary_item_vectors = binary.transpose().tocsr()
        popularity = np.diff(binary_item_vectors.indptr) if self.popularity_alpha else None
        self.graph = adjusted_item_neighbors(
            item_vectors,
            binary_item_vectors,
            query_rows,
            self.max_neighbors,
            self.shrinkage,
            self.min_common_users,
            popularity,
            self.popularity_alpha,
            self.block_size,
            progress,
        )
        return self

    def recommend(self, users, neighbors, k, blocked, fallback_ranking):
        if self.graph is None:
            raise RuntimeError("Call fit first")
        if not 0 < neighbors <= self.max_neighbors:
            raise ValueError("neighbors outside fitted range")
        graph = self.graph if neighbors == self.max_neighbors else truncate_neighbors(self.graph, neighbors)

        def score_user(user, excluded, limit):
            index = self.data.user_index.get(user)
            if index is None:
                return []
            history = self.data.matrix.getrow(index)
            if not history.nnz:
                return []
            links = graph[history.indices].tocsc()
            weights = history.data
            scores = history @ graph
            top = ranked_candidates(scores, self.data.product_ids, excluded, limit)
            result = []
            for product, score in top:
                column = self.data.product_index[product]
                start, end = links.indptr[column:column + 2]
                seeds, similarities = links.indices[start:end], links.data[start:end]
                contributions = weights[seeds] * similarities
                strongest = int(np.argmax(contributions))
                seed = history.indices[seeds[strongest]]
                result.append({"product_id": product, "score": score,
                               "reason_product_id": int(self.data.product_ids[seed]),
                               "reason_user_id": None,
                               "reason_similarity": float(similarities[strongest])})
            return result

        return make_recommendations(users, k, blocked, fallback_ranking, "item_cf", score_user)
