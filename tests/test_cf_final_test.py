"""Check the frozen held-out run with small synthetic inputs only."""
from datetime import datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import polars as pl

from scripts.prepare_model_data import DATA_END, TEST_START, TRAIN_END
from scripts.run_cf_final_test import DEFAULT_MODEL, FROZEN, TEST_FILES, run
from scripts.run_cf_fusion_validation import NAMES, POLICY
from test_cf import history


class FinalTestPipeline(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.reference = self.root / "validation.json"
        self.config = {
            "split": "validation_only", "test_evaluated": False, "target_policy": POLICY,
            "history_end_exclusive_utc": str(TRAIN_END), "answer_end_exclusive_utc": str(TEST_START),
            **{name: value for name, value in FROZEN.items() if name != "item_cf"},
            "model_definition": {"item_cf": FROZEN["item_cf"]},
        }
        self.reference.write_text(json.dumps(self.config), encoding="utf-8")
        self.history = history().with_columns(
            (pl.col("product_id") == 20).cast(pl.Int64).alias("cart_count"),
            pl.lit(datetime(2019, 11, 10)).alias("first_interaction_at"),
            pl.lit(TEST_START - timedelta(seconds=1)).alias("last_interaction_at"),
        )
        self.targets = pl.DataFrame({
            "user_id": [1, 1, 2, 999], "product_id": [20, 30, 30, 555],
            "first_future_purchase_at": [TEST_START] * 4,
            "last_future_purchase_at": [DATA_END - timedelta(seconds=1)] * 4,
        })
        self.history.write_parquet(self.root / TEST_FILES[0])
        self.targets.write_parquet(self.root / TEST_FILES[1])
        pl.DataFrame({"user_id": [1, 2, 999], "known_user": [True, True, False],
                      "activity_segment": ["single_session", "single_session", "new_user"],
                      }).write_parquet(self.root / TEST_FILES[2])

    def test_final_pipeline_and_disjoint_inputs(self):
        output = self.root / "out"
        report = run(self.root, self.reference, output, quiet=True)
        self.assertTrue(report["test_evaluated"])
        self.assertEqual(report["default_model_chosen_on_validation"], DEFAULT_MODEL)
        self.assertEqual(report["frozen_configuration"], FROZEN)
        self.assertEqual(set(report["input_sha256"]), set(TEST_FILES))
        self.assertEqual(set(report["overall_results"]), set(NAMES))
        self.assertEqual(report["selection"]["eligible_target_pairs"], 2)
        self.assertEqual(report["selection"]["removed_prior_purchase_pairs"], 1)
        self.assertEqual(report["selection"]["removed_prior_cart_only_pairs"], 1)
        outcomes = pl.read_parquet(output / "purchase_outcomes.parquet")
        cold = outcomes.filter(pl.col("product_id") == 555).row(0, named=True)
        self.assertTrue(cold["eligible_for_evaluation"])
        self.assertFalse(cold["in_train_catalog"])
        recs = pl.read_parquet(output / "recommendations.parquet")
        self.assertFalse(recs.filter((pl.col("user_id") == 1) & (pl.col("product_id") == 20)).height)
        self.assertNotIn(2, recs["user_id"].to_list())
        self.assertNotIn(555, recs["product_id"].to_list())
        with self.assertRaisesRegex(ValueError, "Output directory exists"):
            run(self.root, self.reference, output, quiet=True)

    def test_rejects_changed_config_before_loading_test_files(self):
        self.reference.write_text(json.dumps({**self.config, "rank_constant": 100}), encoding="utf-8")
        with patch("scripts.run_cf_final_test.load_temporal_inputs") as load:
            with self.assertRaisesRegex(ValueError, "frozen"):
                run(self.root, self.reference, self.root / "out", quiet=True)
            load.assert_not_called()

    def test_rejects_boundary_errors_before_fitting(self):
        bad = self.history.with_columns(pl.lit(TEST_START).alias("last_interaction_at"))
        bad.write_parquet(self.root / TEST_FILES[0])
        with patch("scripts.run_cf_final_test.evaluate_split") as evaluate:
            with self.assertRaisesRegex(ValueError, "cutoff"):
                run(self.root, self.reference, self.root / "out", quiet=True)
            self.history.write_parquet(self.root / TEST_FILES[0])
            self.targets.with_columns(pl.lit(DATA_END).alias("last_future_purchase_at")).write_parquet(self.root / TEST_FILES[1])
            with self.assertRaisesRegex(ValueError, "evaluation window"):
                run(self.root, self.reference, self.root / "out", quiet=True)
            evaluate.assert_not_called()
