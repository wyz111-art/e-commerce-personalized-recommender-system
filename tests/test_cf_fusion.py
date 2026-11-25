"""Check RRF scores, provenance, fallback, and a validation-only run."""
from datetime import datetime
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

from models.cf_common import RECOMMENDATION_SCHEMA
from models.fusion import reciprocal_rank_pool, recommend_fused
from scripts.run_cf_fusion_validation import candidate_diagnostics, run
from test_cf import history


def candidates(source, products):
    return pl.DataFrame([
        (1, product, i, float(i * 100), source if product != 99 else "popularity_fallback",
         7 if source == "item_cf" else None, 8 if source == "user_cf" else None, .5)
        for i, product in enumerate(products, 1)
    ], schema=RECOMMENDATION_SCHEMA, orient="row")


class FusionTests(unittest.TestCase):
    def test_rrf_deduplicates_and_ignores_fallback_and_original_scores(self):
        item = candidates("item_cf", [20, 10, 99])
        user = candidates("user_cf", [30, 10, 99])
        pool = reciprocal_rank_pool(item, user, 60)
        self.assertEqual(pool["product_id"].to_list(), [10, 20, 30])
        best = pool.row(0, named=True)
        self.assertAlmostEqual(best["rrf_score"], 2 / 62)
        self.assertEqual((best["item_cf_rank"], best["user_cf_rank"]), (2, 2))
        self.assertEqual((best["item_cf_seed_product_id"], best["user_cf_peer_user_id"]), (7, 8))
        counts = candidate_diagnostics(pool, {1: {10, 20, 30, 99}, 2: {10}})
        self.assertEqual(counts["answer_pairs_by_source"], {"both": 1, "item_only": 1, "user_only": 1, "neither": 2})
        self.assertAlmostEqual(counts["macro_candidate_recall"]["union"], .375)

    def test_fallback_only_after_fusion_and_respects_exclusions(self):
        pool = reciprocal_rank_pool(candidates("item_cf", [20, 10]), candidates("user_cf", [10]))
        fallback = pl.DataFrame({"product_id": [20, 30, 40], "score": [100., 90., 80.]})
        rows = recommend_fused(pool, [1, 2], 2, {1: {20}}, fallback)
        self.assertEqual(rows["product_id"].to_list(), [10, 30, 20, 30])
        self.assertEqual(rows["source"].to_list(), ["rrf_fusion"] + ["popularity_fallback"] * 3)
        self.assertEqual(rows["item_cf_rank"].to_list(), [2, None, None, None])

    def test_duplicate_candidates_rejected_and_empty_pool_supported(self):
        one = candidates("item_cf", [10])
        empty = reciprocal_rank_pool(one.head(0), one.head(0))
        self.assertTrue(empty.is_empty())
        with self.assertRaises(ValueError):
            reciprocal_rank_pool(pl.concat([one, one]), one.head(0))

    def test_third_channel_keeps_ranks_and_votes_only_once(self):
        item = candidates("item_cf", [20, 10])
        user = candidates("user_cf", [10, 30])
        als = candidates("als", [40, 10, 99])
        pool = reciprocal_rank_pool(item, user, 60, als_candidates=als)
        self.assertEqual(set(pool["product_id"]), {10, 20, 30, 40})
        first = pool.row(0, named=True)
        self.assertEqual(first["product_id"], 10)
        self.assertAlmostEqual(first["rrf_score"], 1 / 61 + 2 / 62)
        self.assertEqual((first["als_rank"], first["als_score"]), (2, 200.))

    def test_pipeline_uses_only_validation_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "input").mkdir()
            h = history().with_columns((pl.col("product_id") == 20).cast(pl.Int64).alias("cart_count"))
            h.write_parquet(root / "input" / "train_interactions.parquet")
            pl.DataFrame({"user_id": [1, 1, 999], "product_id": [20, 30, 30],
                          "first_future_purchase_at": [datetime(2019, 11, 12)] * 3,
                          "last_future_purchase_at": [datetime(2019, 11, 12)] * 3,
                          }).write_parquet(root / "input" / "validation_targets.parquet")
            pl.DataFrame({"user_id": [1, 999], "known_user": [True, False],
                          "activity_segment": ["single_session", "new_user"],
                          }).write_parquet(root / "input" / "validation_user_segments.parquet")
            report = run(root / "input", root / "output", quiet=True)
            self.assertFalse(report["test_evaluated"])
            self.assertEqual(report["selection"]["eligible_target_pairs"], 2)
            self.assertEqual(len(report["overall_results"]), 5)
            recs = pl.read_parquet(root / "output" / "recommendations.parquet")
            self.assertFalse(recs.filter((pl.col("user_id") == 1) & (pl.col("product_id") == 20)).height)
            fused_new = recs.filter((pl.col("model") == "rrf_fusion") & (pl.col("user_id") == 999))
            self.assertTrue((fused_new["source"] == "popularity_fallback").all())
            self.assertTrue((root / "output" / "candidate_pool.parquet").is_file())
            with self.assertRaises(ValueError):
                run(root / "input", root / "output", quiet=True)
