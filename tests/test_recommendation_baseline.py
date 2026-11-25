"""Hand-calculated unit tests for shared recommendation evaluation."""
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl
from models.popularity import PopularityRecommender
from models.metrics import history_exclusions, make_truth, evaluate_users, aggregate_metrics
from scripts.run_popularity_baseline import run


def history():
    return pl.DataFrame({
        "user_id": [1, 1, 2, 2, 3], "product_id": [10, 20, 20, 30, 40],
        "purchase_count": [1, 0, 2, 0, 1], "interaction_score": [7., 4., 8., 2., 5.],
        "last_interaction_at": [datetime(2019, 11, 10)] * 5,
    })


def targets():
    return pl.DataFrame({
        "user_id": [1, 1, 1, 4], "product_id": [10, 20, 50, 20],
        "first_future_purchase_at": [datetime(2019, 11, 12)] * 4,
        "last_future_purchase_at": [datetime(2019, 11, 12)] * 4,
    })


def segments():
    return pl.DataFrame({"user_id": [1, 4], "known_user": [True, False],
                         "activity_segment": ["single_session", "new_user"]})


class BaselineTests(unittest.TestCase):
    def test_ranking_and_exclusions(self):
        h = history()
        model = PopularityRecommender().fit(h)
        self.assertEqual(model.ranking["product_id"].to_list(), [20, 10, 40, 30])
        blocked = history_exclusions(h)
        predictions = model.recommend([1, 4], 2, blocked)
        self.assertEqual(predictions.filter(pl.col("user_id") == 1)["product_id"].to_list(), [20, 40])
        self.assertEqual(predictions.filter(pl.col("user_id") == 4)["product_id"].to_list(), [20, 10])
        self.assertEqual(model.head_products(0.01), {20})

    def test_hand_calculated_metrics_and_cold_items(self):
        h = history()
        blocked = history_exclusions(h)
        truth, selected = make_truth(targets(), blocked)
        self.assertEqual(truth, {1: {20, 50}, 4: {20}})
        self.assertEqual(selected["removed_target_pairs"], 1)
        predictions = PopularityRecommender().fit(h).recommend(truth, 2, blocked)
        info = {r["user_id"]: r for r in segments().to_dicts()}
        rows = evaluate_users(predictions, truth, {10, 20, 30, 40}, info, 2)
        first = rows.filter(pl.col("user_id") == 1).to_dicts()[0]
        self.assertEqual(first["reachable_relevant_items"], 1)
        self.assertEqual(first["recall_at_k"], 0.5)  # unseen product 50 remains in denominator
        self.assertAlmostEqual(first["ndcg_at_k"], 1 / (1 + 1 / math.log2(3)))
        overall = aggregate_metrics(rows, predictions, {10, 20, 30, 40}, {20}, 2)[0]
        self.assertEqual(overall["recall_at_k"], 0.75)
        self.assertEqual(overall["mrr_at_k"], 1)
        self.assertEqual(overall["catalog_coverage"], 0.75)
        self.assertEqual(overall["head_slot_share"], 0.5)

    def test_no_hits_short_lists_and_duplicate_rejection(self):
        truth = {1: {99}}
        info = {1: {"known_user": True, "activity_segment": "single_session"}}
        empty = pl.DataFrame(schema={"user_id": pl.Int64, "product_id": pl.Int64, "rank": pl.Int64})
        row = evaluate_users(empty, truth, {10}, info, 10).to_dicts()[0]
        self.assertEqual((row["recall_at_k"], row["ndcg_at_k"], row["mrr_at_k"]), (0., 0., 0.))
        duplicate = pl.DataFrame({"user_id": [1, 1], "product_id": [10, 10], "rank": [1, 2]})
        with self.assertRaises(ValueError):
            evaluate_users(duplicate, truth, {10}, info, 10)

    def test_policies_and_empty_answer_users(self):
        h = history()
        t = pl.DataFrame({"user_id": [1, 2], "product_id": [10, 50]})
        truth, selection = make_truth(t, history_exclusions(h))
        self.assertEqual(truth, {2: {50}})
        self.assertEqual(selection["removed_empty_label_users"], 1)
        unseen, _ = make_truth(targets(), history_exclusions(h, "unseen"))
        self.assertEqual(unseen, {1: {50}, 4: {20}})
        self.assertEqual(history_exclusions(h, "all"), {})

    def test_pipeline_only_validation_files_required(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            history().write_parquet(data / "train_interactions.parquet")
            targets().write_parquet(data / "validation_targets.parquet")
            segments().write_parquet(data / "validation_user_segments.parquet")
            report = run(data, root / "reports", k=2)
            self.assertFalse(report["test_evaluated"])
            self.assertEqual(len(report["overall_results"]), 2)
            self.assertEqual(report["overall_results"][0]["recall_at_k"], 0.75)
            self.assertTrue((root / "reports" / "RESULTS.md").is_file())
            with self.assertRaises(ValueError):
                run(data, root / "reports", k=2)
            # No test file exists: the pipeline never loads future test inputs.
            h = history().with_columns(pl.lit(datetime(2019, 11, 12)).alias("last_interaction_at"))
            h.write_parquet(data / "train_interactions.parquet")
            with self.assertRaises(ValueError):
                run(data, root / "bad", k=2)


if __name__ == "__main__":
    unittest.main()