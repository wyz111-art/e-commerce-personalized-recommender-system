"""Tests for the optional implicit ALS experiment."""
from importlib.util import find_spec
import unittest

import polars as pl

from models.als import ImplicitALS
from models.cf_common import SparseInteractions
from models.popularity import PopularityRecommender
from test_cf import history


@unittest.skipUnless(find_spec("implicit") is not None, "Install requirements-als.txt for ALS tests")
class ALSTests(unittest.TestCase):
    def test_fit_filters_explicitly_and_can_recommend_viewed_items(self):
        h = history()
        popularity = PopularityRecommender().fit(h).ranking
        als = ImplicitALS(SparseInteractions(h), factors=4, iterations=10, threads=1)
        with self.assertRaises(RuntimeError):
            als.recommend([1], 2, {}, popularity)
        als.fit()
        rows = als.recommend([1, 999], 3, {1: {20}}, popularity)
        first = rows.filter(pl.col("user_id") == 1)
        self.assertNotIn(20, first["product_id"].to_list())
        viewed = first.filter(pl.col("product_id") == 10).row(0, named=True)
        self.assertEqual(viewed["source"], "als")
        self.assertIsNone(viewed["reason_product_id"])
        self.assertTrue((rows.filter(pl.col("user_id") == 999)["source"] == "popularity_fallback").all())
        only_cf = als.recommend([1, 999], 3, {1: {20}}, popularity.head(0), batch_size=1)
        self.assertEqual(set(only_cf["user_id"]), {1})
        self.assertEqual(set(only_cf["source"]), {"als"})
