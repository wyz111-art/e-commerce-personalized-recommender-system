"""Synthetic tests; no database connection and no full dataset required."""
import importlib.util
import math
import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_model_data.py"
spec = importlib.util.spec_from_file_location("prepare_model_data", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def fixture():
    data = [
        ("2019-10-01 10:00:00", 1, 10, "view", "a"),
        ("2019-10-02 10:00:00", 1, 10, "view", "b"),
        ("2019-10-03 10:00:00", 1, 10, "cart", "c"),
        ("2019-10-04 10:00:00", 1, 10, "purchase", "c"),
        ("2019-10-05 10:00:00", 1, 20, "cart", "d"),
        ("2019-10-06 10:00:00", 1, 30, "view", "e"),
        ("2019-10-07 10:00:00", 2, 40, "view", "f"),
        ("2019-11-10 23:59:59.999999", 2, 10, "view", "g"),
        ("2019-11-11 00:00:00", 1, 10, "purchase", "h"),
        ("2019-11-12 10:00:00", 1, 20, "purchase", "h"),
        ("2019-11-13 10:00:00", 1, 30, "purchase", "h"),
        ("2019-11-14 10:00:00", 1, 40, "purchase", "h"),
        ("2019-11-14 11:00:00", 1, 40, "purchase", "h"),
        ("2019-11-15 10:00:00", 3, 40, "purchase", "i"),
        ("2019-11-16 10:00:00", 1, 50, "purchase", "j"),
        ("2019-11-17 10:00:00", 4, 60, "purchase", "k"),
        ("2019-11-18 10:00:00", 5, 70, "view", "l"),
        ("2019-11-20 23:59:59.999999", 1, 80, "view", "m"),
        ("2019-11-21 00:00:00", 5, 70, "purchase", "n"),
        ("2019-11-22 10:00:00", 3, 40, "purchase", "o"),
        ("2019-11-23 10:00:00", 6, 70, "purchase", "p"),
        ("2019-11-24 10:00:00", 1, 90, "purchase", "q"),
        ("2019-11-25 10:00:00", 7, 99, "purchase", "r"),
        ("2019-11-30 23:59:59", 1, 10, "purchase", "s"),
    ]
    return pl.DataFrame({
        "event_at": [datetime.fromisoformat(row[0]) for row in data],
        "user_id": [row[1] for row in data], "product_id": [row[2] for row in data],
        "event_type": [row[3] for row in data],
        "session_key": [f"{row[1]}_{row[4]}" for row in data],
    })


class PrepareModelDataTests(unittest.TestCase):
    def setUp(self):
        self.events = fixture().lazy()
        self.train = self.events.filter(pl.col("event_at") < m.TRAIN_END)
        self.interactions = m.build_interactions(self.train)
        self.users = m.build_user_history_summary(self.train)
        self.products = m.build_product_history_summary(self.train)

    def targets(self, future, history=None):
        if history is None:
            return m.build_targets(future, self.interactions, self.users, self.products)
        return m.build_targets(future, m.build_interactions(history),
                               m.build_user_history_summary(history),
                               m.build_product_history_summary(history))

    def test_exact_cutoffs_and_score(self):
        self.assertEqual(self.interactions["event_count"].sum(), 8)
        self.assertLess(self.interactions["last_interaction_at"].max(), m.TRAIN_END)
        row = self.interactions.filter((pl.col("user_id") == 1) & (pl.col("product_id") == 10)).to_dicts()[0]
        self.assertEqual((row["view_count"], row["cart_count"], row["purchase_count"]), (2, 1, 1))
        self.assertAlmostEqual(row["interaction_score"], math.log1p(2) + 3 + 5)
        self.assertNotIn(80, self.interactions["product_id"].to_list())

    def test_target_relationships_and_duplicate_purchases(self):
        future = self.events.filter((pl.col("event_at") >= m.TRAIN_END) & (pl.col("event_at") < m.TEST_START))
        targets = self.targets(future)
        self.assertEqual(targets.height, 7)
        self.assertEqual(targets["future_purchase_event_count"].sum(), 8)
        pairs = {(r["user_id"], r["product_id"]): r for r in targets.to_dicts()}
        self.assertEqual(pairs[1, 10]["prior_user_product_segment"], "prior_purchase")
        self.assertEqual(pairs[1, 20]["prior_user_product_segment"], "prior_cart_only")
        self.assertEqual(pairs[1, 30]["prior_user_product_segment"], "prior_view_only")
        self.assertEqual(pairs[1, 40]["prior_user_product_segment"], "no_prior_user_product_interaction")
        self.assertEqual(pairs[1, 40]["future_purchase_event_count"], 2)
        self.assertEqual(pairs[1, 40]["future_purchase_session_count"], 1)
        self.assertEqual(pairs[3, 40]["cold_start_segment"], "new_user_known_product")
        self.assertEqual(pairs[1, 50]["cold_start_segment"], "known_user_new_product")
        self.assertEqual(pairs[4, 60]["cold_start_segment"], "new_user_new_product")
        self.assertFalse(pairs[1, 10]["is_new_purchase_for_user"])
        self.assertTrue(pairs[1, 40]["is_unseen_to_user"])
        segments = m.build_target_user_segments(targets, self.users)
        self.assertEqual(segments.height, 3)
        new = segments.filter(pl.col("user_id") == 3).to_dicts()[0]
        self.assertFalse(new["known_user"])
        self.assertEqual((new["event_count"], new["activity_segment"]), (0, "new_user"))

    def test_test_history_includes_validation_not_future(self):
        history = self.events.filter(pl.col("event_at") < m.TEST_START)
        future = self.events.filter(pl.col("event_at") >= m.TEST_START)
        pairs = {(r["user_id"], r["product_id"]): r for r in self.targets(future, history).to_dicts()}
        self.assertEqual(pairs[5, 70]["cold_start_segment"], "known_user_known_product")
        self.assertEqual(pairs[5, 70]["prior_user_product_segment"], "prior_view_only")
        self.assertEqual(pairs[3, 40]["prior_user_product_segment"], "prior_purchase")
        self.assertEqual(pairs[6, 70]["cold_start_segment"], "new_user_known_product")
        self.assertEqual(pairs[1, 90]["cold_start_segment"], "known_user_new_product")
        self.assertEqual(pairs[7, 99]["cold_start_segment"], "new_user_new_product")

    def test_empty_history_and_empty_targets(self):
        empty = self.events.filter(pl.lit(False))
        empty_targets = self.targets(empty)
        self.assertEqual(empty_targets.height, 0)
        self.assertEqual(m.build_target_user_segments(empty_targets, self.users).height, 0)
        targets = self.targets(self.events.filter(pl.col("event_at") >= m.TEST_START), empty)
        self.assertTrue((targets["cold_start_segment"] == "new_user_new_product").all())

    def test_full_pipeline_roundtrip_and_no_overwrite(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "events.parquet"
            fixture().write_parquet(source)
            original = m.sha256_file(source)
            report = m.prepare(source, root / "outputs")
            self.assertEqual(m.sha256_file(source), original)
            self.assertEqual(report["validation"]["purchase_events"], 8)
            self.assertEqual(report["validation"]["purchase_sessions"], 4)
            self.assertEqual(report["test"]["purchase_events"], 6)
            self.assertTrue(report["test"]["checks_passed"])
            self.assertEqual(len(list((root / "outputs").glob("*.parquet"))), 10)
            with self.assertRaises(ValueError):
                m.prepare(source, root / "outputs")
            with self.assertRaises(ValueError):
                m.prepare(source, root / "bad" , view_weight=float("nan"))


if __name__ == "__main__":
    unittest.main()