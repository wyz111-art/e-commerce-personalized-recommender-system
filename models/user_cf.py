"""Weighted cosine User-CF; exact neighbors are queried only for requested users."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from .cf_common import (SparseInteractions, normalize_rows, cosine_neighbors,
                        ranked_candidates, make_recommendations)


class UserCF:
    def __init__(self, data: SparseInteractions, max_neighbors=100, block_size=64):
        self.data = data
        self.max_neighbors = max_neighbors
        self.block_size = block_size
        self.neighbor_rows = {}
        self.graph = None
        self.normalized = None

    def fit(self):
        self.normalized = normalize_rows(self.data.matrix)
        return self

    def prepare_neighbors(self, users, progress=None):
        if self.normalized is None:
            raise RuntimeError("Call fit first")
        known = sorted(set(users) & self.data.user_index.keys())
        query = np.asarray([self.data.user_index[user] for user in known], dtype=np.int64)
        self.graph = cosine_neighbors(self.normalized, query, self.max_neighbors,
                                      self.block_size, progress)
        self.neighbor_rows = {user: position for position, user in enumerate(known)}
        return self

    def recommend(self, users, neighbors, k, blocked, fallback_ranking):
        users = list(users)
        if not 0 < neighbors <= self.max_neighbors:
            raise ValueError("neighbors outside fitted range")
        if self.graph is None or any(
            user in self.data.user_index and user not in self.neighbor_rows for user in users
        ):
            self.prepare_neighbors(users)

        def score_user(user, excluded, limit):
            position = self.neighbor_rows.get(user)
            if position is None:
                return []
            start = self.graph.indptr[position]
            end = min(start + neighbors, self.graph.indptr[position + 1])
            neighbor_ids = self.graph.indices[start:end]
            similarities = self.graph.data[start:end]
            if not len(neighbor_ids):
                return []
            histories = self.data.matrix[neighbor_ids]
            denominator = float(similarities.sum())
            scores = (sparse.csr_matrix(similarities.reshape(1, -1)) @ histories).tocsr()
            scores.data /= denominator
            top = ranked_candidates(scores, self.data.product_ids, excluded, limit)
            histories = histories.tocsc()
            result = []
            for product, score in top:
                column = self.data.product_index[product]
                a, b = histories.indptr[column:column + 2]
                peers = histories.indices[a:b]
                contributions = similarities[peers] * histories.data[a:b]
                strongest = peers[int(np.argmax(contributions))]
                result.append({"product_id": product, "score": score,
                               "reason_product_id": None,
                               "reason_user_id": int(self.data.user_ids[neighbor_ids[strongest]]),
                               "reason_similarity": float(similarities[strongest])})
            return result

        return make_recommendations(users, k, blocked, fallback_ranking, "user_cf", score_user)