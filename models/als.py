"""Implicit-feedback ALS with bounded-memory scoring and shared exclusions."""
from __future__ import annotations

import numpy as np

from .cf_common import make_recommendations, select_top


class ImplicitALS:
    def __init__(self, data, *, factors=32, regularization=.05, iterations=20, confidence_scale=5., seed=42,
                 threads=2):
        if min(factors, iterations, threads) < 1 or not all(
            np.isfinite(x) and x > 0 for x in (regularization, confidence_scale)
        ):
            raise ValueError("ALS factors, iterations, threads, regularization and confidence scale must be positive")
        self.data = data
        self.options = dict(factors=factors, regularization=regularization, iterations=iterations,
                            confidence_scale=confidence_scale, seed=seed, threads=threads)
        self.model = None

    def fit(self):
        from implicit.cpu.als import AlternatingLeastSquares
        from threadpoolctl import threadpool_limits

        opt = self.options
        confidence = self.data.matrix.copy()
        confidence.data = 1. + opt["confidence_scale"] * confidence.data
        with threadpool_limits(limits=1, user_api="blas"):
            model = AlternatingLeastSquares(factors=opt["factors"], regularization=opt["regularization"],
                                             iterations=opt["iterations"], random_state=opt["seed"],
                                             num_threads=opt["threads"], alpha=1.)
            model.fit(confidence, show_progress=False)
        if not (np.isfinite(model.user_factors).all() and np.isfinite(model.item_factors).all()):
            raise ValueError("Nonfinite ALS factors")
        self.model = model
        return self

    def recommend(self, users, k, blocked, fallback_ranking, *, batch_size=32):
        from threadpoolctl import threadpool_limits

        if self.model is None:
            raise RuntimeError("Call fit first")
        if min(k, batch_size) < 1:
            raise ValueError("k and batch_size must be positive")
        users = sorted(set(users))
        known = [(user, self.data.user_index[user]) for user in users if user in self.data.user_index]
        products = np.flatnonzero(np.diff(self.data.matrix.tocsc().indptr))
        factors = self.model.item_factors[products].T
        positions = {int(self.data.product_ids[p]): i for i, p in enumerate(products)}
        candidates = {}
        with threadpool_limits(limits=1, user_api="blas"):
            for start in range(0, len(known), batch_size):
                batch = known[start:start + batch_size]
                scores = self.model.user_factors[[index for _, index in batch]] @ factors
                for (user, _), values in zip(batch, scores):
                    denied = [positions[item] for item in blocked.get(user, ()) if item in positions]
                    values[denied] = -np.inf
                    ids, top_scores = select_top(products, values, k)
                    candidates[user] = [
                        {"product_id": int(self.data.product_ids[pos]), "score": float(score),
                         "reason_product_id": None, "reason_user_id": None, "reason_similarity": None}
                        for pos, score in zip(ids, top_scores)
                    ]
        return make_recommendations(users, k, blocked, fallback_ranking, "als",
                                    lambda user, _blocked, _k: candidates.get(user, []))
