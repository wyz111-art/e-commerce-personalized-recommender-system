"""Dense reference checks on tiny examples for sparse Item-CF and User-CF."""
from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
from scipy import sparse
from models.cf_common import (SparseInteractions, normalize_rows, cosine_neighbors,
                              select_top, truncate_neighbors)
from models.item_cf import ItemCF
from models.user_cf import UserCF
from models.popularity import PopularityRecommender


def history():
    return pl.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 4],
        "product_id": [10, 20, 10, 30, 20, 40, 50],
        "interaction_score": [1., 1., 1., 1., 1., 1., 1.],
        "purchase_count": [0, 0, 0, 1, 0, 1, 0],
        "last_interaction_at": [datetime(2019, 11, 10)] * 7,
    })


class CFTests(unittest.TestCase):
    def setUp(self):
        self.data = SparseInteractions(history())
        self.fallback = PopularityRecommender().fit(history()).ranking

    def test_sparse_item_graph_matches_dense_reference(self):
        model = ItemCF(self.data, max_neighbors=10, block_size=2).fit()
        dense = self.data.matrix.toarray().T
        norms = np.linalg.norm(dense, axis=1)
        expected = (dense / norms[:, None]) @ (dense / norms[:, None]).T
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(model.graph.toarray(), expected, atol=1e-6)
        self.assertLessEqual(model.graph.nnz, len(self.data.product_ids) * 10)
        self.assertEqual(model.graph.diagonal().sum(), 0)

    def test_adjusted_item_graph_matches_dense_reference(self):
        rows = [(1, 10, 1.), (1, 20, 2.), (1, 30, 1.), (1, 50, 1.),
                (2, 10, 3.), (2, 20, 1.), (3, 10, 1.), (3, 40, 1.), (4, 20, 1.)]
        data = SparseInteractions(pl.DataFrame(rows, schema=["user_id", "product_id", "interaction_score"],
                                                orient="row"))
        dense = data.matrix.toarray().astype(np.float64)
        binary = (dense > 0).astype(int)
        supports = binary.T @ binary
        iuf = 1 / np.log1p(binary.sum(axis=1))
        vectors = (dense * np.sqrt(iuf[:, None])).T
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        expected = (vectors @ vectors.T) * (supports / (supports + 5.))
        expected /= binary.sum(axis=0) ** .3
        expected[supports < 2] = 0
        np.fill_diagonal(expected, 0)
        model = ItemCF(data, 4, 2, shrinkage=5, min_common_users=2, iuf=True, popularity_alpha=.3).fit()
        np.testing.assert_allclose(model.graph.toarray(), expected, atol=1e-6)
        self.assertEqual(model.graph.nnz, 2)
        base = ItemCF(data, 4, 2).fit().graph
        shrunk = ItemCF(data, 4, 2, shrinkage=5).fit().graph
        self.assertAlmostEqual(shrunk[0, 1] / base[0, 1], 2 / 7, places=6)
        self.assertAlmostEqual(shrunk[0, 2] / base[0, 2], 1 / 6, places=6)
        penalty = ItemCF(data, 4, 2, popularity_alpha=1).fit().graph
        self.assertAlmostEqual(penalty[0, 2], base[0, 2], places=6)
        self.assertAlmostEqual(penalty[2, 0], base[2, 0] / 3, places=6)
        self.assertEqual(ItemCF(data, 1, 2, min_common_users=2).fit().graph.nnz, 2)

    def test_iuf_favors_low_activity_common_user_and_keeps_explanation(self):
        rows = [(1, 10), (1, 20), (1, 40), (1, 50), (2, 10), (2, 30)]
        data = SparseInteractions(pl.DataFrame(rows, schema=["user_id", "product_id"], orient="row")
                                  .with_columns(pl.lit(1.).alias("interaction_score")))
        base = ItemCF(data, 5).fit()
        model = ItemCF(data, 5, iuf=True).fit()
        self.assertAlmostEqual(base.graph[0, 1], base.graph[0, 2])
        self.assertLess(model.graph[0, 1], model.graph[0, 2])
        rec = model.recommend([1], 5, 1, {1: {10, 20, 40, 50}}, self.fallback).row(0, named=True)
        self.assertEqual((rec["source"], rec["product_id"], rec["reason_product_id"]), ("item_cf", 30, 10))
        self.assertAlmostEqual(rec["score"], rec["reason_similarity"])

    def test_invalid_adjustment_parameters(self):
        for options in ({"shrinkage": -1}, {"shrinkage": float("nan")}, {"popularity_alpha": float("inf")},
                        {"popularity_alpha": -.1}, {"min_common_users": 0}, {"min_common_users": 1.5},
                        {"iuf": "false"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ItemCF(self.data, **options)

    def test_top_ties_are_deterministic_and_truncation_is_by_rank(self):
        ids, scores = select_top(np.array([7, 2, 9, 1]), np.array([.5, .5, .9, .5]), 2)
        self.assertEqual(ids.tolist(), [9, 1])
        g = sparse.csr_matrix((np.array([.8, .5]), np.array([4, 1]), np.array([0, 2])), shape=(1, 5))
        result = truncate_neighbors(g, 1)
        self.assertEqual(result.indices.tolist(), [4])
        self.assertEqual(select_top(np.array([1, 2]), np.array([1., 0.]), 2, 1)[0].size, 0)

    def test_item_recommendations_and_reasons(self):
        model = ItemCF(self.data, max_neighbors=10, block_size=2).fit()
        result = model.recommend([1], 10, 2, {1: {10, 20}}, self.fallback)
        self.assertEqual(result["product_id"].to_list(), [30, 40])
        self.assertTrue((result["source"] == "item_cf").all())
        self.assertEqual(result["reason_product_id"].to_list(), [10, 20])
        np.testing.assert_allclose(result["score"].to_numpy(), [1 / np.sqrt(2)] * 2, rtol=1e-5)

    def test_user_recommendations_and_dense_reference(self):
        model = UserCF(self.data, max_neighbors=10, block_size=2).fit().prepare_neighbors([1])
        self.assertEqual(model.graph.shape, (1, 4))
        np.testing.assert_allclose(model.graph.toarray()[0], [0, .5, .5, 0], atol=1e-6)
        result = model.recommend([1], 10, 2, {1: {10, 20}}, self.fallback)
        self.assertEqual(result["product_id"].to_list(), [30, 40])
        self.assertEqual(result["reason_user_id"].to_list(), [2, 3])
        self.assertTrue((result["source"] == "user_cf").all())
        np.testing.assert_allclose(result["score"].to_numpy(), [.5, .5], rtol=1e-5)

    def test_new_isolated_users_and_insufficient_candidates_fall_back(self):
        for model in [ItemCF(self.data, 10, 2).fit(), UserCF(self.data, 10, 2).fit()]:
            result = model.recommend([4, 999], 10, 2, {4: {50}}, self.fallback)
            self.assertEqual(result.height, 4)
            self.assertTrue((result["source"] == "popularity_fallback").all())
            self.assertEqual(result.filter(pl.col("user_id") == 999)["product_id"].to_list(), [30, 40])
            partial = model.recommend([1], 1, 5, {1: {10, 20}}, self.fallback)
            self.assertEqual(partial["product_id"].n_unique(), partial.height)
            self.assertNotIn(10, partial["product_id"].to_list())
            self.assertNotIn(20, partial["product_id"].to_list())

    def test_zero_vectors_and_invalid_inputs(self):
        data = history().with_columns(pl.lit(0.).alias("interaction_score"))
        empty_scores = SparseInteractions(data)
        self.assertEqual(normalize_rows(empty_scores.matrix).nnz, 0)
        for model in [ItemCF(empty_scores, 2).fit(), UserCF(empty_scores, 2).fit()]:
            result = model.recommend([1], 2, 2, {}, self.fallback)
            self.assertTrue((result["source"] == "popularity_fallback").all())
        with self.assertRaises(ValueError):
            SparseInteractions(pl.concat([history(), history().head(1)]))
        with self.assertRaises(ValueError):
            SparseInteractions(history().with_columns(pl.lit(float("nan")).alias("interaction_score")))
        with self.assertRaises(ValueError):
            ItemCF(self.data, 2).fit().recommend([1], 3, 10, {}, self.fallback)

    def test_weighted_cosine_graph_with_multiple_blocks(self):
        h = history().with_columns(pl.Series("interaction_score", [2., 1., 3., .5, 4., 2., .8]))
        data = SparseInteractions(h)
        for matrix in [data.matrix, data.matrix.transpose().tocsr()]:
            normalized = normalize_rows(matrix)
            graph = cosine_neighbors(normalized, np.arange(matrix.shape[0]), 100, block_size=1)
            dense = normalized.toarray()
            expected = dense @ dense.T
            np.fill_diagonal(expected, 0)
            np.testing.assert_allclose(graph.toarray(), expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
