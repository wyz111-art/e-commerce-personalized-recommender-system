"""End-to-end validation comparison with synthetic files only."""
from datetime import datetime
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl
from scripts.run_cf_validation import run
from scripts.tune_item_cf_validation import Config, run as tune
from test_cf import history


class CFValidationPipelineTests(unittest.TestCase):
    def test_cart_and_purchase_filter_with_adjusted_item_preset(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            stamp = datetime(2019, 11, 10)
            pl.DataFrame({
                "user_id": [1, 1, 1, 2, 2, 2, 3, 3],
                "product_id": [10, 20, 30, 20, 30, 40, 20, 30],
                "cart_count": [1, 0, 1, 1, 0, 0, 0, 0],
                "purchase_count": [1, 0, 0, 0, 0, 1, 0, 0],
                "interaction_score": [8., 1., 5., 4., 1., 8., 1., 1.],
                "last_interaction_at": [stamp] * 8,
            }).write_parquet(data / "train_interactions.parquet")
            stamp = datetime(2019, 11, 12)
            pl.DataFrame({
                "user_id": [1, 1, 1, 2, 999], "product_id": [10, 20, 30, 20, 30],
                "first_future_purchase_at": [stamp] * 5,
                "last_future_purchase_at": [stamp] * 5,
            }).write_parquet(data / "validation_targets.parquet")
            pl.DataFrame({
                "user_id": [1, 2, 999], "known_user": [True, True, False],
                "activity_segment": ["single_session", "single_session", "new_user"],
            }).write_parquet(data / "validation_user_segments.parquet")

            output = root / "discovery"
            report = run(data, output, neighbors=[3], k=2, block_size=2, quiet=True,
                         target_policy="exclude_cart_and_purchase", item_preset="shrinkage_iuf")
            self.assertFalse(report["test_evaluated"])
            self.assertEqual(report["item_cf_parameters"]["shrinkage"], 15.)
            self.assertEqual(report["item_cf_parameters"]["min_common_users"], 2)
            self.assertEqual(report["selection"]["eligible_target_pairs"], 2)
            self.assertEqual(report["selection"]["removed_prior_purchase_pairs"], 1)
            self.assertEqual(report["selection"]["removed_additional_prior_cart_pairs"], 2)
            self.assertEqual(report["selection"]["removed_additional_prior_cart_users"], 1)
            recs = pl.read_parquet(output / "recommendations.parquet")
            self.assertEqual(set(recs["user_id"]), {1, 999})
            mine = recs.filter(pl.col("user_id") == 1)
            self.assertEqual(set(mine["product_id"]), {20, 40})
            explained = mine.filter(pl.col("source") == "item_cf").row(0, named=True)
            self.assertEqual((explained["product_id"], explained["reason_product_id"]), (20, 30))

    def test_no_test_files_needed_and_recommendations_explainable(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            history().write_parquet(data / "train_interactions.parquet")
            pl.DataFrame({
                "user_id": [1, 1, 4, 999], "product_id": [30, 40, 50, 30],
                "first_future_purchase_at": [datetime(2019, 11, 12)] * 4,
                "last_future_purchase_at": [datetime(2019, 11, 12)] * 4,
            }).write_parquet(data / "validation_targets.parquet")
            pl.DataFrame({
                "user_id": [1, 4, 999], "known_user": [True, True, False],
                "activity_segment": ["single_session", "single_session", "new_user"],
            }).write_parquet(data / "validation_user_segments.parquet")
            report = run(data, root / "reports", neighbors=[1, 2], k=2, block_size=2, quiet=True)
            self.assertFalse(report["test_evaluated"])
            self.assertEqual(len(report["overall_results"]), 6)
            self.assertEqual(report["matrix"]["users"], 4)
            self.assertEqual(report["matrix"]["products"], 5)
            recs = pl.read_parquet(root / "reports" / "recommendations.parquet")
            self.assertEqual(recs.height, 36)
            self.assertTrue(recs.filter(pl.col("source") == "item_cf")["reason_product_id"].is_not_null().all())
            self.assertTrue(recs.filter(pl.col("source") == "user_cf")["reason_user_id"].is_not_null().all())
            new_recs = recs.filter((pl.col("user_id") == 999) & pl.col("model").str.starts_with("item_cf"))
            self.assertTrue((new_recs["source"] == "popularity_fallback").all())
            metrics = pl.read_csv(root / "reports" / "metrics_summary.csv")
            self.assertIn("known_users", metrics["user_group"].to_list())
            self.assertTrue((root / "reports" / "RESULTS.md").is_file())
            with self.assertRaises(ValueError):
                run(data, root / "reports", neighbors=[1, 2], k=2, quiet=True)

            tuning_dir = root / "tuning"
            tuned = tune(data, tuning_dir, neighbors=2, k=2, block_size=2, quiet=True)
            self.assertFalse(tuned["test_evaluated"])
            self.assertEqual(tuned["input_sha256"], report["input_sha256"])
            self.assertEqual(tuned["selection"], report["selection"])
            old = next(row for row in tuned["overall_results"] if row["model"] == Config().name)
            previous = next(row for row in report["overall_results"] if row["model"] == "item_cf_n2")
            for metric in ("recall_at_k", "ndcg_at_k", "catalog_coverage", "fallback_slot_share"):
                self.assertEqual(old[metric], previous[metric])
            stages = {row["stage"] for row in tuned["runs"]}
            self.assertEqual(stages, {"baseline", "support_shrinkage", "iuf", "popularity_penalty"})
            self.assertIn(len(tuned["runs"]), (14, 15))
            best_recs = pl.read_parquet(tuning_dir / "best_item_cf_recommendations.parquet")
            self.assertEqual(best_recs["model"].unique().to_list(), [tuned["best_item_cf"]["model"]])
            self.assertTrue(best_recs.filter(pl.col("source") == "item_cf")["reason_product_id"].is_not_null().all())
            self.assertTrue((tuning_dir / "RESULTS.md").is_file())
            with self.assertRaises(ValueError):
                tune(data, tuning_dir, quiet=True)


if __name__ == "__main__":
    unittest.main()
